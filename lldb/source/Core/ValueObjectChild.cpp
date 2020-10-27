//===-- ValueObjectChild.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ValueObjectChild.h"

#include "lldb/Core/Value.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/Flags.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"

#include <functional>
#include <memory>
#include <vector>

#include <stdio.h>
#include <string.h>

using namespace lldb_private;

ValueObjectChild::ValueObjectChild(
    ValueObject &parent, const CompilerType &compiler_type,
    ConstString name, uint64_t byte_size, int32_t byte_offset,
    uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
    bool is_base_class, bool is_deref_of_parent,
    AddressType child_ptr_or_ref_addr_type, uint64_t language_flags)
    : ValueObject(parent), m_compiler_type(compiler_type),
      m_byte_size(byte_size), m_byte_offset(byte_offset),
      m_bitfield_bit_size(bitfield_bit_size),
      m_bitfield_bit_offset(bitfield_bit_offset),
      m_is_base_class(is_base_class), m_is_deref_of_parent(is_deref_of_parent),
      m_can_update_with_invalid_exe_ctx() {
  m_name = name;
  SetAddressTypeOfChildren(child_ptr_or_ref_addr_type);
  SetLanguageFlags(language_flags);
}

ValueObjectChild::~ValueObjectChild() {}

lldb::ValueType ValueObjectChild::GetValueType() const {
  return m_parent->GetValueType();
}

size_t ValueObjectChild::CalculateNumChildren(uint32_t max) {
  ExecutionContext exe_ctx(GetExecutionContextRef());
  auto children_count = GetCompilerType().GetNumChildren(true, &exe_ctx);
  return children_count <= max ? children_count : max;
}

static void AdjustForBitfieldness(ConstString &name,
                                  uint8_t bitfield_bit_size) {
  if (name && bitfield_bit_size)
    name.SetString(llvm::formatv("{0}:{1}", name, bitfield_bit_size).str());
}

ConstString ValueObjectChild::GetTypeName() {
  if (m_type_name.IsEmpty()) {
    m_type_name = GetCompilerType().GetTypeName();
    AdjustForBitfieldness(m_type_name, m_bitfield_bit_size);
  }
  return m_type_name;
}

ConstString ValueObjectChild::GetQualifiedTypeName() {
  ConstString qualified_name = GetCompilerType().GetTypeName();
  AdjustForBitfieldness(qualified_name, m_bitfield_bit_size);
  return qualified_name;
}

ConstString ValueObjectChild::GetDisplayTypeName() {
  ConstString display_name = GetCompilerType().GetDisplayTypeName();
  AdjustForBitfieldness(display_name, m_bitfield_bit_size);
  return display_name;
}

LazyBool ValueObjectChild::CanUpdateWithInvalidExecutionContext() {
  if (m_can_update_with_invalid_exe_ctx.hasValue())
    return m_can_update_with_invalid_exe_ctx.getValue();
  if (m_parent) {
    ValueObject *opinionated_parent =
        m_parent->FollowParentChain([](ValueObject *valobj) -> bool {
          return (valobj->CanUpdateWithInvalidExecutionContext() ==
                  eLazyBoolCalculate);
        });
    if (opinionated_parent)
      return (m_can_update_with_invalid_exe_ctx =
                  opinionated_parent->CanUpdateWithInvalidExecutionContext())
          .getValue();
  }
  return (m_can_update_with_invalid_exe_ctx =
              this->ValueObject::CanUpdateWithInvalidExecutionContext())
      .getValue();
}

bool ValueObjectChild::UpdateValue() {
  m_error.Clear();
  SetValueIsValid(false);
  ValueObject *parent = m_parent;
  if (parent) {
    if (parent->UpdateValueIfNeeded(false)) {
      m_value.SetCompilerType(GetCompilerType());

      CompilerType parent_type(parent->GetCompilerType());
      // Copy the parent scalar value and the scalar value type
      m_value.GetScalar() = parent->GetValue().GetScalar();
      Value::ValueType value_type = parent->GetValue().GetValueType();
      m_value.SetValueType(value_type);

      Flags parent_type_flags(parent_type.GetTypeInfo());
      const bool is_instance_ptr_base =
          ((m_is_base_class) &&
           (parent_type_flags.AnySet(lldb::eTypeInstanceIsPointer)));

      if (parent->GetCompilerType().ShouldTreatScalarValueAsAddress()) {
        lldb::addr_t addr = parent->GetPointerValue();
        m_value.GetScalar() = addr;

        if (addr == LLDB_INVALID_ADDRESS) {
          m_error.SetErrorString("parent address is invalid.");
        } else if (addr == 0) {
          m_error.SetErrorString("parent is NULL");
        } else {
          m_value.GetScalar() += m_byte_offset;
          AddressType addr_type = parent->GetAddressTypeOfChildren();

          switch (addr_type) {
          case eAddressTypeFile: {
            lldb::ProcessSP process_sp(GetProcessSP());
            if (process_sp && process_sp->IsAlive())
              m_value.SetValueType(Value::eValueTypeLoadAddress);
            else
              m_value.SetValueType(Value::eValueTypeFileAddress);
          } break;
          case eAddressTypeLoad:
            m_value.SetValueType(is_instance_ptr_base
                                     ? Value::eValueTypeScalar
                                     : Value::eValueTypeLoadAddress);
            break;
          case eAddressTypeHost:
            m_value.SetValueType(Value::eValueTypeHostAddress);
            break;
          case eAddressTypeInvalid:
            // TODO: does this make sense?
            m_value.SetValueType(Value::eValueTypeScalar);
            break;
          }
        }
      } else {
        switch (value_type) {
        case Value::eValueTypeLoadAddress:
        case Value::eValueTypeFileAddress:
        case Value::eValueTypeHostAddress: {
          lldb::addr_t addr =
              m_value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
          if (addr == LLDB_INVALID_ADDRESS) {
            m_error.SetErrorString("parent address is invalid.");
          } else if (addr == 0) {
            m_error.SetErrorString("parent is NULL");
          } else {
            // Set this object's scalar value to the address of its value by
            // adding its byte offset to the parent address
            m_value.GetScalar() += GetByteOffset();

            // If a bitfield doesn't fit into the child_byte_size'd
            // window at child_byte_offset, move the window forward
            // until it fits.  The problem here is that Value has no
            // notion of bitfields and thus the Value's DataExtractor
            // is sized like the bitfields CompilerType; a sequence of
            // bitfields, however, can be larger than their underlying
            // type.
            if (m_bitfield_bit_offset) {
              const bool thread_and_frame_only_if_stopped = true;
              ExecutionContext exe_ctx(GetExecutionContextRef().Lock(
                  thread_and_frame_only_if_stopped));
              if (auto type_bit_size = GetCompilerType().GetBitSize(
                      exe_ctx.GetBestExecutionContextScope())) {
                uint64_t bitfield_end =
                    m_bitfield_bit_size + m_bitfield_bit_offset;
                if (bitfield_end > *type_bit_size) {
                  uint64_t overhang_bytes =
                      (bitfield_end - *type_bit_size + 7) / 8;
                  m_value.GetScalar() += overhang_bytes;
                  m_bitfield_bit_offset -= overhang_bytes * 8;
                }
              }
            }
          }
        } break;

        case Value::eValueTypeScalar:
          // try to extract the child value from the parent's scalar value
          {
            Scalar scalar(m_value.GetScalar());
            if (m_bitfield_bit_size)
              scalar.ExtractBitfield(m_bitfield_bit_size,
                                     m_bitfield_bit_offset);
            else
              scalar.ExtractBitfield(8 * m_byte_size, 8 * m_byte_offset);
            m_value.GetScalar() = scalar;
          }
          break;
        default:
          m_error.SetErrorString("parent has invalid value.");
          break;
        }
      }

      if (m_error.Success()) {
        const bool thread_and_frame_only_if_stopped = true;
        ExecutionContext exe_ctx(
            GetExecutionContextRef().Lock(thread_and_frame_only_if_stopped));
        if (GetCompilerType().GetTypeInfo() & lldb::eTypeHasValue) {
          Value &value = is_instance_ptr_base ? m_parent->GetValue() : m_value;
          m_error =
              value.GetValueAsData(&exe_ctx, m_data, GetModule().get());
        } else {
          m_error.Clear(); // No value so nothing to read...
        }
      }

    } else {
      m_error.SetErrorStringWithFormat("parent failed to evaluate: %s",
                                       parent->GetError().AsCString());
    }
  } else {
    m_error.SetErrorString("ValueObjectChild has a NULL parent ValueObject.");
  }

  return m_error.Success();
}

bool ValueObjectChild::IsInScope() {
  ValueObject *root(GetRoot());
  if (root)
    return root->IsInScope();
  return false;
}
