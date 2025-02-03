//===-- Value.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_VALUE_H
#define LLDB_CORE_VALUE_H

#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-types.h"

#include "llvm/ADT/APInt.h"

#include <vector>

#include <cstdint>
#include <cstring>

namespace lldb_private {
class DataExtractor;
class ExecutionContext;
class Module;
class Stream;
class Type;
class Variable;
}

namespace lldb_private {

class Value {
public:
  /// Type that describes Value::m_value.
  enum class ValueType {
    Invalid = -1,
    // m_value contains:
    /// A raw scalar value.
    Scalar = 0,
    /// A file address value.
    FileAddress,
    /// A load address value.
    LoadAddress,
    /// A host address value (for memory in the process that < A is
    /// using liblldb).
    HostAddress
  };

  /// Type that describes Value::m_context.
  enum class ContextType {
    // m_context contains:
    /// Undefined.
    Invalid = -1,
    /// RegisterInfo * (can be a scalar or a vector register).
    RegisterInfo = 0,
    /// lldb_private::Type *.
    LLDBType,
    /// lldb_private::Variable *.
    Variable
  };

  Value();
  Value(const Scalar &scalar);
  Value(const void *bytes, int len);
  Value(const Value &rhs);

  void SetBytes(const void *bytes, int len);

  void AppendBytes(const void *bytes, int len);

  Value &operator=(const Value &rhs);

  const CompilerType &GetCompilerType();

  void SetCompilerType(const CompilerType &compiler_type);

  ValueType GetValueType() const;

  AddressType GetValueAddressType() const;

  ContextType GetContextType() const { return m_context_type; }

  void SetValueType(ValueType value_type) { m_value_type = value_type; }

  void ClearContext() {
    m_context = nullptr;
    m_context_type = ContextType::Invalid;
  }

  void SetContext(ContextType context_type, void *p) {
    m_context_type = context_type;
    m_context = p;
    if (m_context_type == ContextType::RegisterInfo) {
      RegisterInfo *reg_info = GetRegisterInfo();
      if (reg_info->encoding == lldb::eEncodingVector)
        SetValueType(ValueType::Scalar);
    }
  }

  RegisterInfo *GetRegisterInfo() const;

  Type *GetType();

  Scalar &ResolveValue(ExecutionContext *exe_ctx, Module *module = nullptr);

  /// See comment on m_scalar to understand what GetScalar returns.
  const Scalar &GetScalar() const { return m_value; }

  /// See comment on m_scalar to understand what GetScalar returns.
  Scalar &GetScalar() { return m_value; }

  size_t ResizeData(size_t len);

  size_t AppendDataToHostBuffer(const Value &rhs);

  DataBufferHeap &GetBuffer() { return m_data_buffer; }

  const DataBufferHeap &GetBuffer() const { return m_data_buffer; }

  bool ValueOf(ExecutionContext *exe_ctx);

  Variable *GetVariable();

  void Dump(Stream *strm);

  lldb::Format GetValueDefaultFormat();

  uint64_t GetValueByteSize(Status *error_ptr, ExecutionContext *exe_ctx);

  Status GetValueAsData(ExecutionContext *exe_ctx, DataExtractor &data,
                        Module *module); // Can be nullptr

  static const char *GetValueTypeAsCString(ValueType context_type);

  static const char *GetContextTypeAsCString(ContextType context_type);

  /// Convert this value's file address to a load address, if possible.
  void ConvertToLoadAddress(Module *module, Target *target);

  bool GetData(DataExtractor &data);

  void Clear();

  static ValueType GetValueTypeFromAddressType(AddressType address_type);

protected:
  /// Represents a value, which can be a scalar, a load address, a file address,
  /// or a host address.
  ///
  /// The interpretation of `m_value` depends on `m_value_type`:
  /// - Scalar: `m_value` contains the scalar value.
  /// - Load Address: `m_value` contains the load address.
  /// - File Address: `m_value` contains the file address.
  /// - Host Address: `m_value` contains a pointer to the start of the buffer in
  ///    host memory.
  ///   Currently, this can point to either:
  ///     - The `m_data_buffer` of this Value instance (e.g., in DWARF
  ///     computations).
  ///     - The `m_data` of a Value Object containing this Value.
  // TODO: the GetScalar() API relies on knowledge not codified by the type
  // system, making it hard to understand and easy to misuse.
  // - Separate the scalar from the variable that contains the address (be it a
  //   load, file or host address).
  // - Rename GetScalar() to something more indicative to what the scalar is,
  //   like GetScalarOrAddress() for example.
  // - Split GetScalar() into two functions, GetScalar() and GetAddress(), which
  //   verify (or assert) what m_value_type is to make sure users of the class are
  //   querying the right thing.
  // TODO: It's confusing to point to multiple possible buffers when the
  // ValueType is a host address. Value should probably always own its buffer.
  // Perhaps as a shared pointer with a copy on write system if the same buffer
  // can be shared by multiple classes.
  Scalar m_value;
  CompilerType m_compiler_type;
  void *m_context = nullptr;
  ValueType m_value_type = ValueType::Scalar;
  ContextType m_context_type = ContextType::Invalid;
  DataBufferHeap m_data_buffer;
};

class ValueList {
public:
  ValueList() = default;
  ~ValueList() = default;

  ValueList(const ValueList &rhs) = default;
  ValueList &operator=(const ValueList &rhs) = default;

  // void InsertValue (Value *value, size_t idx);
  void PushValue(const Value &value);

  size_t GetSize();
  Value *GetValueAtIndex(size_t idx);
  void Clear();

private:
  typedef std::vector<Value> collection;

  collection m_values;
};

} // namespace lldb_private

#endif // LLDB_CORE_VALUE_H
