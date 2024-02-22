//===-- DILEval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DILEval.h"

#include <memory>

#include "clang/Basic/TokenKinds.h"
#include "lldb/Core/DILAST.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

namespace lldb_private {

template <typename T>
bool Compare(BinaryOpKind kind, const T& l, const T& r) {
  switch (kind) {
    case BinaryOpKind::EQ:
      return l == r;
    case BinaryOpKind::NE:
      return l != r;
    case BinaryOpKind::LT:
      return l < r;
    case BinaryOpKind::LE:
      return l <= r;
    case BinaryOpKind::GT:
      return l > r;
    case BinaryOpKind::GE:
      return l >= r;

    default:
      assert(false && "invalid ast: invalid comparison operation");
      return false;
  }
}

/*
static lldb::ValueObjectSP CreateValueObjectFromBytes (lldb::TargetSP target_sp,
                                                 const void* bytes,
                                                 CompilerType type) {
  ExecutionContext exe_ctx(
      ExecutionContextRef(ExecutionContext(target_sp.get(), false)));
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target_sp.get()))
    byte_size = temp.value();
  lldb::DataExtractorSP data_sp =
      std::make_shared<DataExtractor> (
          bytes, byte_size,
          target_sp->GetArchitecture().GetByteOrder(),
          static_cast<uint8_t>(
              target_sp->GetArchitecture().GetAddressByteSize()));
  lldb::ValueObjectSP value =
      ValueObject::CreateValueObjectFromData("result", *data_sp, exe_ctx,
                                             type);
  return value;

}

static lldb::ValueObjectSP CreateValueObjectFromBytes (lldb::TargetSP target,
                                                 const void* bytes,
                                                 lldb::BasicType type) {
  CompilerType target_type;
  if (target) {
    for (auto type_system_sp : target->GetScratchTypeSystems())
      if (auto compiler_type = type_system_sp->GetBasicTypeFromAST(type)) {
        target_type = compiler_type;
        break;
      }
  }
  return CreateValueObjectFromBytes(target, bytes, target_type);
}

static lldb::ValueObjectSP CreateValueObjectFromAPInt (lldb::TargetSP target,
                                                 const llvm::APInt &v,
                                                 CompilerType type) {
  return CreateValueObjectFromBytes(target, v.getRawData(), type);
}

static lldb::ValueObjectSP CreateValueObjectFromAPFloat (lldb::TargetSP target,
                                                   const llvm::APFloat& v,
                                                   CompilerType type) {
  return CreateValueObjectFromAPInt(target, v.bitcastToAPInt(), type);
}

static lldb::ValueObjectSP CreateValueObjectFromPointer (lldb::TargetSP target,
                                                uintptr_t addr,
                                                CompilerType type) {
  return CreateValueObjectFromBytes(target, &addr, type);
}

static lldb::ValueObjectSP CreateValueObjectFromBool (lldb::TargetSP target,
                                                bool value) {
  return CreateValueObjectFromBytes(target, &value, lldb::eBasicTypeBool);
}


static lldb::ValueObjectSP CreateValueObjectFromNullptr(lldb::TargetSP target,
                                                  CompilerType type) {
  assert(type.IsNullPtrType() && "target type must be nullptr");
  uintptr_t zero = 0;
  return CreateValueObjectFromBytes(target, &zero, type);
}
*/
/*
static llvm::APFloat CreateAPFloatFromAPSInt(const llvm::APSInt& value,
                                             lldb::BasicType basic_type) {
  switch (basic_type) {
    case lldb::eBasicTypeFloat:
      return llvm::APFloat(value.isSigned()
                               ? llvm::APIntOps::RoundSignedAPIntToFloat(value)
                               : llvm::APIntOps::RoundAPIntToFloat(value));
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble:
      return llvm::APFloat(value.isSigned()
                               ? llvm::APIntOps::RoundSignedAPIntToDouble(value)
                               : llvm::APIntOps::RoundAPIntToDouble(value));
    default:
      return llvm::APFloat(NAN);
  }
}

static llvm::APFloat CreateAPFloatFromAPFloat(llvm::APFloat value,
                                              lldb::BasicType basic_type) {
  switch (basic_type) {
    case lldb::eBasicTypeFloat: {
      bool loses_info;
      value.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
      return value;
    }
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble: {
      bool loses_info;
      value.convert(llvm::APFloat::IEEEdouble(),
                    llvm::APFloat::rmNearestTiesToEven, &loses_info);
      return value;
    }
    default:
      return llvm::APFloat(NAN);
  }
}
*/
/*
static uint64_t GetByteSize(CompilerType type, lldb::TargetSP target) {
  //ExecutionContext exe_ctx(
  //    ExecutionContextRef(ExecutionContext(target.get(), false)));
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    byte_size = temp.value();

  return byte_size;
}
*/

/*
static llvm::APSInt GetInteger(lldb::ValueObjectSP value_sp)
{
  lldb::TargetSP target = value_sp->GetTargetSP();
  uint64_t byte_size = 0;
  if (auto temp = value_sp->GetCompilerType().GetByteSize(target.get()))
    byte_size = temp.value();

  unsigned bit_width = static_cast<unsigned>(byte_size * CHAR_BIT);
  lldb::ValueObjectSP value(DILGetSPWithLock(value_sp));
  bool success = true;
  uint64_t fail_value = 0;
  uint64_t ret_val = value->GetValueAsUnsigned(fail_value, &success);
  uint64_t new_value = fail_value;
  if (success)
    new_value = ret_val;
  bool is_signed = value->GetCompilerType().IsSigned();

  return llvm::APSInt(llvm::APInt(bit_width, new_value, is_signed), !is_signed);
}
*/

/*
static llvm::APFloat GetFloat(lldb::ValueObjectSP value) {
  lldb::BasicType basic_type =
      value->GetCompilerType().GetCanonicalType().GetBasicTypeEnumeration();
  lldb::DataExtractorSP data_sp(new DataExtractor());
  Status error;

  switch (basic_type) {
    case lldb::eBasicTypeFloat: {
      float v = 0;
      value->GetData(*data_sp, error);
      assert (error.Success() && "Unable to read float data from value");

      lldb::offset_t offset = 0;
      uint32_t old_offset = offset;
      void *ok = nullptr;
      ok = data_sp->GetU8(&offset, (void *) &v, sizeof(float));
      assert(offset != old_offset && ok != nullptr && "unable to read data");

      return llvm::APFloat(v);
    }
    case lldb::eBasicTypeDouble:
      // No way to get more precision at the moment.
    case lldb::eBasicTypeLongDouble: {
      double v = 0;
      value->GetData(*data_sp, error);
      assert (error.Success() && "Unable to read long double data from value");

      lldb::offset_t offset = 0;
      uint32_t old_offset = offset;
      void *ok = nullptr;
      ok = data_sp->GetU8(&offset, (void *) &v, sizeof(double));
      assert(offset != old_offset && ok != nullptr && "unable to read data");

      return llvm::APFloat(v);
    }
    default:
      return llvm::APFloat(NAN);
  }
}
*/

/*
static uint64_t GetUInt64(lldb::ValueObjectSP value_sp) {
  // GetValueAsUnsigned performs overflow according to the underlying type. For
  // example, if the underlying type is `int32_t` and the value is `-1`,
  // GetValueAsUnsigned will return 4294967295.
  lldb::ValueObjectSP value(DILGetSPWithLock(value_sp));
  return value->GetCompilerType().IsSigned()
      ? value->GetValueAsSigned(0)
      : value->GetValueAsUnsigned(0);
}
*/

/*
static bool GetBool(lldb::ValueObjectSP value) {
  CompilerType val_type = value->GetCompilerType();
  if (val_type.IsInteger() || val_type.IsUnscopedEnumerationType() ||
      val_type.IsPointerType()) {
    return GetInteger(value).getBoolValue();
  }
  if (val_type.IsFloat()) {
    return GetFloat(value).isNonZero();
  }
  if (val_type.IsArrayType()) {
    lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
    lldb::ValueObjectSP new_val =
        ValueObject::ValueObject::CreateValueObjectFromAddress(
            value_sp->GetName().GetStringRef(),
            value_sp->GetAddressOf(),
            value_sp->GetExecutionContextRef(),
            val_type);
    //return GetUInt64(new_val) != 0;
    return new_val->GetValueAsUnsigned(0) != 0;
  }
  return false;
}
*/

/*
static lldb::ValueObjectSP Clone(lldb::ValueObjectSP val) {
  lldb::DataExtractorSP data_sp(new DataExtractor());
  Status error;
  val->GetData(*data_sp, error);
  if (error.Success()) {
    Status ignore;
    void *ok = nullptr;
    auto raw_data = std::make_unique<uint8_t[]>(data_sp->GetByteSize());
    lldb::offset_t offset = 0;
    uint32_t old_offset = offset;
    size_t size = data_sp->GetByteSize();
    ok = data_sp->GetU8(&offset, raw_data.get(), size);
    if ((offset == old_offset) || (ok == nullptr)) {
      ignore.SetErrorString("Clone: unable to read data");
      return lldb::ValueObjectSP();
    }
    return ValueObject::CreateValueObjectFromBytes(
        val->GetTargetSP(), raw_data.get(), val->GetCompilerType());
  } else {
    return lldb::ValueObjectSP();
  }
}
*/

/*
static void Update(lldb::ValueObjectSP val, const llvm::APInt& v) {
  lldb::TargetSP target = val->GetTargetSP();
  uint64_t byte_size = 0;
  if (auto temp = val->GetCompilerType().GetByteSize(target.get()))
    byte_size = temp.value();

  assert(v.getBitWidth() == byte_size * CHAR_BIT &&
         "illegal argument: new value should be of the same size");

  lldb::DataExtractorSP data_sp;
  Status error;
  data_sp->SetData(v.getRawData(), byte_size,
                   target->GetArchitecture().GetByteOrder());
  data_sp->SetAddressByteSize(
      static_cast<uint8_t>(target->GetArchitecture().GetAddressByteSize()));
  val->SetData(*data_sp, error);
}
*/

/*
static void Update(lldb::ValueObjectSP val, lldb::ValueObjectSP val2) {
  CompilerType val2_type = val2->GetCompilerType();
  assert((val2_type.IsInteger() || val2_type.IsFloat() ||
          val2_type.IsPointerType()) &&
         "illegal argument: new value should be of the same size");

  if (val2_type.IsInteger()) {
    val.UpdateIntegerValue(val2->GetValueAsAPSInt());
  } else if (val2_type.IsFloat()) {
    val.UpdateIntegerValue(val2->GetValueAsFloat().bitcastToAPInt());
  } else if (val2_type.IsPointerType()) {
    val.UpdateIntegerValue(llvm::APInt(64, val2->GetValueAsUnsigned(0)));
    //Update(val, llvm::APInt(64, GetUInt64(val2)));
  }
}
*/

static lldb::ValueObjectSP EvaluateArithmeticOpInteger(lldb::TargetSP target,
                                                       BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs,
                                                       CompilerType rtype)
{
  assert(lhs->GetCompilerType().IsInteger() &&
         rhs->GetCompilerType().IsInteger() &&
         "invalid ast: both operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr ||
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must have the same type");

  auto wrap = [target, rtype](auto value) {
    return ValueObject::CreateValueObjectFromAPInt(target, value, rtype);
  };

  auto l = lhs->GetValueAsAPSInt();
  auto r = rhs->GetValueAsAPSInt();

  switch (kind) {
    case BinaryOpKind::Add:
      return wrap(l + r);
    case BinaryOpKind::Sub:
      return wrap(l - r);
    case BinaryOpKind::Div:
      return wrap(l / r);
    case BinaryOpKind::Mul:
      return wrap(l * r);
    case BinaryOpKind::Rem:
      return wrap(l % r);
    case BinaryOpKind::And:
      return wrap(l & r);
    case BinaryOpKind::Or:
      return wrap(l | r);
    case BinaryOpKind::Xor:
      return wrap(l ^ r);
    case BinaryOpKind::Shl:
      return wrap(l.shl(r));
    case BinaryOpKind::Shr:
      // Apply arithmetic shift on signed values and logical shift operation
      // on unsigned values.
      return wrap(l.isSigned() ? l.ashr(r) : l.lshr(r));

    default:
      assert(false && "invalid ast: invalid arithmetic operation");
      return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOpFloat(lldb::TargetSP target,
                                                     BinaryOpKind kind,
                                                     lldb::ValueObjectSP lhs,
                                                     lldb::ValueObjectSP rhs,
                                                     CompilerType rtype) {
  assert((lhs->GetCompilerType().IsFloat() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be floats and have the same type");

  auto wrap = [target, rtype](auto value) {
    return ValueObject::CreateValueObjectFromAPFloat(target, value, rtype);
  };

  auto l = lhs->GetValueAsFloat();
  auto r = rhs->GetValueAsFloat();

  switch (kind) {
    case BinaryOpKind::Add:
      return wrap(l + r);
    case BinaryOpKind::Sub:
      return wrap(l - r);
    case BinaryOpKind::Div:
      return wrap(l / r);
    case BinaryOpKind::Mul:
      return wrap(l * r);

    default:
      assert(false && "invalid ast: invalid arithmetic operation");
      return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOp(lldb::TargetSP target,
                                                BinaryOpKind kind,
                                                lldb::ValueObjectSP lhs,
                                                lldb::ValueObjectSP rhs,
                                                CompilerType rtype) {
  assert((rtype.IsInteger() || rtype.IsFloat()) &&
         "invalid ast: result type must either integer or floating point");

  // Evaluate arithmetic operation for two integral values.
  if (rtype.IsInteger()) {
    return EvaluateArithmeticOpInteger(target, kind, lhs, rhs, rtype);
  }

  // Evaluate arithmetic operation for two floating point values.
  if (rtype.IsFloat()) {
    return EvaluateArithmeticOpFloat(target, kind, lhs, rhs, rtype);
  }

  return lldb::ValueObjectSP();
}

static bool IsInvalidDivisionByMinusOne(lldb::ValueObjectSP lhs_sp,
                                        lldb::ValueObjectSP rhs_sp)
{
  assert(lhs_sp->GetCompilerType().IsInteger() &&
         rhs_sp->GetCompilerType().IsInteger() && "operands should be integers");

  lldb::ValueObjectSP rhs(DILGetSPWithLock(rhs_sp));
  lldb::ValueObjectSP lhs(DILGetSPWithLock(lhs_sp));
  // The result type should be signed integer.
  auto basic_type =
      rhs->GetCompilerType().GetCanonicalType().GetBasicTypeEnumeration();
  if (basic_type != lldb::eBasicTypeInt && basic_type != lldb::eBasicTypeLong &&
      basic_type != lldb::eBasicTypeLongLong) {
    return false;
  }

  // The RHS should be equal to -1.
  if (rhs->GetValueAsSigned(0) != -1) {
    return false;
  }

  // The LHS should be equal to the minimum value the result type can hold.
  uint64_t byte_size = 0;
  if (auto temp = rhs->GetCompilerType().GetByteSize(rhs->GetTargetSP().get()))
    byte_size = temp.value();
  auto bit_size = byte_size * CHAR_BIT;
  return lhs->GetValueAsSigned(0) + (1LLU << (bit_size - 1)) == 0;
}

static lldb::ValueObjectSP EvaluateMemberOf(lldb::ValueObjectSP value,
                                            const std::vector<uint32_t>& path,
                                            bool use_synthetic,
                                            bool is_dynamic) {
  // The given `value` can be a pointer, but GetChildAtIndex works for pointers
  // too, so we don't need to dereference it explicitely. This also avoid having
  // an "ephemeral" parent lldb::ValueObjectSP, representing the dereferenced
  // value.
  lldb::ValueObjectSP member_val_sp = value;
  // Objects from the standard library (e.g. containers, smart pointers) have
  // synthetic children (e.g. stored values for containers, wrapped object for
  // smart pointers), but the indexes in `member_index()` array refer to the
  // actual type members.
  lldb::DynamicValueType use_dynamic = (!is_dynamic)
                                       ? lldb::eNoDynamicValues
                                       : lldb::eDynamicDontRunTarget;
  lldb::ValueObjectSP member_val(DILGetSPWithLock(member_val_sp, use_dynamic,
                                                  use_synthetic));
  for (uint32_t idx : path) {
    // Force static value, otherwise we can end up with the "real" type.
    member_val = member_val->GetChildAtIndex(idx, /*can_create*/ true);
  }
  if (!member_val && is_dynamic) {
    lldb::ValueObjectSP dyn_val_sp = value->GetDynamicValue(use_dynamic);
    if (dyn_val_sp) {
      lldb::ValueObjectSP dyn_member_val(DILGetSPWithLock(dyn_val_sp,
                                                          use_dynamic,
                                                          use_synthetic));
      for (uint32_t idx : path) {
        dyn_member_val = dyn_member_val->GetChildAtIndex(idx, true);
      }
      member_val = dyn_member_val;
    }
  }
  assert(member_val && "invalid ast: invalid member access");

  // If value is a reference, derefernce it to get to the underlying type. All
  // operations on a reference should be actually operations on the referent.
  Status error;
  if (member_val->GetCompilerType().IsReferenceType()) {
    member_val = member_val->Dereference(error);
    assert(member_val && error.Success() && "unable to dereference member val");
  }

  return member_val;
}

/*
static lldb::addr_t GetLoadAddress(lldb::ValueObjectSP inner_value_sp) {
  lldb::addr_t addr_value = LLDB_INVALID_ADDRESS;
  lldb::TargetSP target_sp(inner_value_sp->GetTargetSP());
  lldb::ValueObjectSP inner_value(DILGetSPWithLock(inner_value_sp));
  if (target_sp) {
    const bool scalar_is_load_address = true;
    AddressType addr_type;
    addr_value = inner_value->GetAddressOf(scalar_is_load_address, &addr_type);
    if (addr_type == eAddressTypeFile) {
      lldb::ModuleSP module_sp(inner_value->GetModule());
      if (!module_sp)
        addr_value = LLDB_INVALID_ADDRESS;
      else {
        Address tmp_addr;
        module_sp->ResolveFileAddress(addr_value, tmp_addr);
        addr_value = tmp_addr.GetLoadAddress(target_sp.get());
      }
    } else if (addr_type == eAddressTypeHost ||
               addr_type == eAddressTypeHost)
      addr_value = LLDB_INVALID_ADDRESS;
  }
  return addr_value;
}
*/

/*
static lldb::ValueObjectSP CastDerivedToBaseType(
    lldb::TargetSP target,
    lldb::ValueObjectSP value,
    CompilerType type,
    const std::vector<uint32_t>& idx)
{
  assert((type.IsPointerType() || type.IsReferenceType()) &&
         "invalid ast: target type should be a pointer or a reference");
  assert(!idx.empty() && "invalid ast: children sequence should be non-empty");

  // The `value` can be a pointer, but GetChildAtIndex works for pointers too.
  bool prefer_synthetic_value = false;
  lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues;
  lldb::ValueObjectSP inner_value(DILGetSPWithLock(value, use_dynamic,
                                                   prefer_synthetic_value));
  for (const uint32_t i : idx) {
    // Force static value, otherwise we can end up with the "real" type.
    inner_value = value->GetChildAtIndex(i, **can_create_synthetic** false);
  }

  // At this point type of `inner_value` should be the dereferenced target type.
  CompilerType inner_value_type = inner_value->GetCompilerType();
  if (type.IsPointerType()) {
    assert(inner_value_type.CompareTypes(type.GetPointeeType()) &&
           "casted value doesn't match the desired type");

    uintptr_t addr = inner_value->GetLoadAddress();
    return ValueObject::CreateValueObjectFromPointer(target, addr, type);
  }

  // At this point the target type should be a reference.
  assert(inner_value_type.CompareTypes(type.GetNonReferenceType()) &&
         "casted value doesn't match the desired type");

  lldb::ValueObjectSP inner_value_sp(DILGetSPWithLock(inner_value));
  return lldb::ValueObjectSP(inner_value_sp->Cast(type.GetNonReferenceType()));
}
*/

/*
static lldb::ValueObjectSP CastBaseToDerivedType(lldb::TargetSP target,
                                                 lldb::ValueObjectSP value,
                                                 CompilerType type,
                                                 uint64_t offset)
{
  assert((type.IsPointerType() || type.IsReferenceType()) &&
         "invalid ast: target type should be a pointer or a reference");

  auto pointer_type = type.IsPointerType()
                          ? type
                          : type.GetNonReferenceType().GetPointerType();

  //uintptr_t addr = type.IsPointerType() ? GetUInt64(value)
  uintptr_t addr = type.IsPointerType() ? value->GetValueAsUnsigned(0)
                                        : value->GetLoadAddress();

  value = ValueObject::CreateValueObjectFromPointer(target, addr - offset,
                                                    pointer_type);

  if (type.IsPointerType()) {
    return value;
  }

  // At this point the target type is a reference. Since `value` is a pointer,
  // it has to be dereferenced.
  Status error;
  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  return value_sp->Dereference(error);
}
*/

static std::string FormatDiagnostics(clang::SourceManager& sm,
                                     const std::string& message,
                                     clang::SourceLocation loc,
                                     ErrorCode code)
{
  const char *ecode_names[7] = {
    "kOK", "kInvalidExpressionSyntax", "kInvalidNumericLiteral",
    "kInvalidOperandType", "kUndeclaredIdentifier", "kNotImplemented",
    "kUnknown"};

  // Translate ErrorCode
  llvm::StringRef error_code = ecode_names[(int)code];

  // Get the source buffer and the location of the current token.
  llvm::StringRef text = sm.getBufferData(sm.getFileID(loc));
  size_t loc_offset = sm.getCharacterData(loc) - text.data();

  // Look for the start of the line.
  size_t line_start = text.rfind('\n', loc_offset);
  line_start = line_start == llvm::StringRef::npos ? 0 : line_start + 1;

  // Look for the end of the line.
  size_t line_end = text.find('\n', loc_offset);
  line_end = line_end == llvm::StringRef::npos ? text.size() : line_end;

  // Get a view of the current line in the source code and the position of the
  // diagnostics pointer.
  llvm::StringRef line = text.slice(line_start, line_end);
  int32_t arrow = sm.getPresumedColumnNumber(loc);

  // Calculate the padding in case we point outside of the expression (this can
  // happen if the parser expected something, but got EOF).
  size_t expr_rpad = std::max(0, arrow - static_cast<int32_t>(line.size()));
  size_t arrow_rpad = std::max(0, static_cast<int32_t>(line.size()) - arrow);

  return llvm::formatv("{0}: {1}:{2}\n{3}\n{4}", loc.printToString(sm),
                       error_code, message,
                       llvm::fmt_pad(line, 0, expr_rpad),
                       llvm::fmt_pad("^", arrow - 1, arrow_rpad));
}

void SetUbStatus(Status& error, ErrorCode code) {
  llvm::StringRef err_str;
  switch ((int) code) {
    case (int) ErrorCode::kUBDivisionByZero:
      err_str ="Error: Division by zero detected.";
      break;
    case (int) ErrorCode::kUBDivisionByMinusOne:
      // If "a / b" isn't representable in its result type, then results of
      // "a / b" and "a % b" are undefined behaviour. This happens when "a"
      // is equal to the minimum value of the result type and "b" is equal
      // to -1.
      err_str ="Error: Invalid division by negative one  detected.";
      break;
    case (int) ErrorCode::kUBInvalidCast:
      err_str ="Error: Invalid type cast detected.";
      break;
    case (int) ErrorCode::kUBInvalidShift:
      err_str ="Error: Invalid shift detected.";
      break;
    case (int) ErrorCode::kUBNullPtrArithmetic:
      err_str ="Error: Attempt to perform arithmetic with null ptr  detected.";
      break;
    case (int) ErrorCode::kUBInvalidPtrDiff:
      err_str ="Error: Attempt to perform invalid ptr arithmetic detected.";
      break;
    default:
      err_str ="Error: Unknown undefined behavior error.";
      break;
  }
  error.SetError((lldb::ValueType)code, lldb::ErrorType::eErrorTypeGeneric);
  error.SetErrorString(err_str);
}

/*
static lldb::ValueObjectSP CastScalarToBasicType(lldb::TargetSP target,
                                                 lldb::ValueObjectSP value,
                                                 CompilerType type,
                                                 Status& error)
{
  assert(type.IsScalarType() && "target type must be an scalar");
  assert(value->GetCompilerType().IsScalarType()
         && "argument must be a scalar");

  if (type.IsBoolean()) {
    if (value->GetCompilerType().IsInteger()) {
      return ValueObject::CreateValueObjectFromBool(target,
                                                    value->GetValueAsUnsigned(0) != 0);
      //GetUInt64(value) != 0);
    }
    if (value->GetCompilerType().IsFloat()) {
      return ValueObject::CreateValueObjectFromBool(target,
                                                    !value->GetValueAsFloat().isZero());
    }
  }
  if (type.IsInteger()) {
    if (value->GetCompilerType().IsInteger()) {
      uint64_t byte_size = 0;
      if (auto temp = type.GetByteSize(target.get()))
        byte_size = temp.value();
      llvm::APSInt ext =
          value->GetValueAsAPSInt().extOrTrunc(byte_size * CHAR_BIT);
      return ValueObject::CreateValueObjectFromAPInt(target, ext, type);
    }
    if (value->GetCompilerType().IsFloat()) {
      uint64_t byte_size = 0;
      if (auto temp = type.GetByteSize(target.get()))
        byte_size = temp.value();
      llvm::APSInt integer(byte_size * CHAR_BIT,
                           !type.IsSigned());
      bool is_exact;
      llvm::APFloatBase::opStatus status =
          value->GetValueAsFloat().convertToInteger(
              integer, llvm::APFloat::rmTowardZero, &is_exact);

      // Casting floating point values that are out of bounds of the target type
      // is undefined behaviour.
      if (status & llvm::APFloatBase::opInvalidOp) {
        SetUbStatus(error, ErrorCode::kUBInvalidCast);
      }

      return ValueObject::CreateValueObjectFromAPInt(target, integer, type);
    }
  }
  if (type.IsFloat()) {
    if (value->GetCompilerType().IsInteger()) {
      llvm::APFloat f = CreateAPFloatFromAPSInt(
          value->GetValueAsAPSInt(),
          type.GetCanonicalType().GetBasicTypeEnumeration());
      return ValueObject::CreateValueObjectFromAPFloat(target, f, type);
    }
    if (value->GetCompilerType().IsFloat()) {
      llvm::APFloat f = CreateAPFloatFromAPFloat(
          value->GetValueAsFloat(),
          type.GetCanonicalType().GetBasicTypeEnumeration());
      return ValueObject::CreateValueObjectFromAPFloat(target, f, type);
    }
  }
  assert(false && "invalid target type: must be a scalar");
  return lldb::ValueObjectSP();
}
*/

/*
static lldb::ValueObjectSP CastEnumToBasicType(lldb::TargetSP target,
                                               lldb::ValueObjectSP val,
                                               CompilerType type)
{
  assert(type.IsScalarType() && "target type must be a scalar");
  assert(val->GetCompilerType().IsEnumerationType()
         && "argument must be an enum");

  if (type.IsBoolean()) {
    //return ValueObject::CreateValueObjectFromBool(target, GetUInt64(val) != 0);
    return ValueObject::CreateValueObjectFromBool(target,
                                                  val->GetValueAsUnsigned(0)
                                                  != 0);
  }

  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    byte_size = temp.value();
  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      val->GetValueAsAPSInt().extOrTrunc(byte_size * CHAR_BIT);

  if (type.IsInteger()) {
    return ValueObject::CreateValueObjectFromAPInt(target, ext, type);
  }
  if (type.IsFloat()) {
    llvm::APFloat f =
        CreateAPFloatFromAPSInt(
            ext,
            type.GetCanonicalType().GetBasicTypeEnumeration());
    return ValueObject::CreateValueObjectFromAPFloat(target, f, type);
  }
  assert(false && "invalid target type: must be a scalar");
  return lldb::ValueObjectSP();
}
*/

/*
static lldb::ValueObjectSP CastPointerToBasicType(lldb::TargetSP target,
                                                  lldb::ValueObjectSP val,
                                                  CompilerType type)
{
  uint64_t type_byte_size = 0;
  uint64_t val_byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    type_byte_size = temp.value();
  if (auto temp = val->GetCompilerType().GetByteSize(target.get()))
    val_byte_size = temp.value();
  assert(type.IsInteger() && "target type must be an integer");
  assert((type.IsBoolean() ||  type_byte_size >= val_byte_size)
         && "target type cannot be smaller than the pointer type");

  if (type.IsBoolean()) {
    //return ValueObject::CreateValueObjectFromBool(target, GetUInt64(val) != 0);
    return ValueObject::CreateValueObjectFromBool(target,
                                                  val->GetValueAsUnsigned(0)
                                                  != 0);
  }

  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      val->GetValueAsAPSInt().extOrTrunc(type_byte_size * CHAR_BIT);
  return ValueObject::CreateValueObjectFromAPInt(target, ext, type);
}
*/

/*
static lldb::ValueObjectSP CastIntegerOrEnumToEnumType(lldb::TargetSP target,
                                                       lldb::ValueObjectSP val,
                                                       CompilerType type)
{
  assert(type.IsEnumerationType() && "target type must be an enum");
  assert((val->GetCompilerType().IsInteger()
          || val->GetCompilerType().IsEnumerationType())
         && "argument must be an integer or an enum");
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    byte_size = temp.value();

  // Get the value as APSInt and extend or truncate it to the requested size.
  llvm::APSInt ext =
      val->GetValueAsAPSInt().extOrTrunc(byte_size * CHAR_BIT);
  return ValueObject::CreateValueObjectFromAPInt(target, ext, type);
}
*/

/*
static lldb::ValueObjectSP CastFloatToEnumType(lldb::TargetSP target,
                                               lldb::ValueObjectSP val,
                                               CompilerType type,
                                               Status& error)
{
  assert(type.IsEnumerationType() && "target type must be an enum");
  assert(val->GetCompilerType().IsFloat() && "argument must be a float");

  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(target.get()))
    byte_size = temp.value();
  llvm::APSInt integer(byte_size * CHAR_BIT, !type.IsSigned());
  bool is_exact;

  llvm::APFloatBase::opStatus status =
      val->GetValueAsFloat().convertToInteger(integer,
                                                llvm::APFloat::rmTowardZero,
                                                &is_exact);

  // Casting floating point values that are out of bounds of the target type
  // is undefined behaviour.
  if (status & llvm::APFloatBase::opInvalidOp) {
    SetUbStatus(error, ErrorCode::kUBInvalidCast);
  }

  return ValueObject::CreateValueObjectFromAPInt(target, integer, type);
}
*/

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm)
    : m_target(std::move(target)), m_sm(std::move(sm))
{
  m_default_dynamic = lldb::eNoDynamicValues;
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::DynamicValueType use_dynamic)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_default_dynamic(use_dynamic) {}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::ValueObjectSP scope)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_scope(std::move(scope))
{
  m_default_dynamic = lldb::eNoDynamicValues;
  // If `m_scope` is a reference, dereference it. All operations on a reference
  // should be operations on the referent.
  if (m_scope->GetCompilerType().IsValid() &&
      m_scope->GetCompilerType().IsReferenceType()) {
    Status error;
    m_scope = m_scope->Dereference(error);
  }
}

void DILInterpreter::SetContextVars(
    std::unordered_map<std::string, lldb::ValueObjectSP> context_vars) {
  m_context_vars = std::move(context_vars);
}

lldb::ValueObjectSP DILInterpreter::DILEval(const DILASTNode* tree,
                                            lldb::TargetSP target_sp,
                                            Status& error)
{
  m_error.Clear();
  // Evaluate an AST.
  DILEvalNode(tree);
  // Set the error.
  error = m_error;
  // Return the computed result. If there was an error, it will be invalid.
  return m_result;
}

lldb::ValueObjectSP DILInterpreter::DILEvalNode(const DILASTNode* node,
                                             FlowAnalysis* flow) {
  // Set up the evaluation context for the current node.
  m_flow_analysis_chain.push_back(flow);
  // Traverse an AST pointed by the `node`.
  node->Accept(this);
  // Cleanup the context.
  m_flow_analysis_chain.pop_back();
  // Return the computed value for convenience. The caller is responsible for
  // checking if an error occured during the evaluation.
  return m_result;
}

void DILInterpreter::SetError(ErrorCode code, std::string error,
                              clang::SourceLocation loc) {
  assert(m_error.Success() && "interpreter can error only once");
  m_error.SetErrorString(
      FormatDiagnostics(m_sm->GetSourceManager(), error, loc, code));
}

void DILInterpreter::Visit(const DILErrorNode* node) {
  // The AST is not valid.
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const LiteralNode* node) {
  struct {
    lldb::ValueObjectSP operator()(llvm::APInt val) {
      return ValueObject::CreateValueObjectFromAPInt(target, val, type);
    }
    lldb::ValueObjectSP operator()(llvm::APFloat val) {
      return ValueObject::CreateValueObjectFromAPFloat(target, val, type);
    }
    lldb::ValueObjectSP operator()(bool val) {
      return ValueObject::CreateValueObjectFromBool(target, val);
    }
    lldb::ValueObjectSP operator()(const std::vector<char>& val) {
      return ValueObject::CreateValueObjectFromBytes(
          target, reinterpret_cast<const void*>(val.data()), type);
    }

    lldb::TargetSP target;
    CompilerType type;
  } visitor{m_target, node->result_type()};
  m_result = std::visit(visitor, node->value());
}

void DILInterpreter::Visit(const IdentifierNode* node) {
  auto identifier = static_cast<const IdentifierInfo&>(node->info());

  lldb::ValueObjectSP val;
  lldb::TargetSP target_sp;
  Status error;
  switch (identifier.kind()) {
    using Kind = IdentifierInfo::Kind;
    case Kind::kValue:
      val = identifier.value();
      target_sp = val->GetTargetSP();
      assert(target_sp && target_sp->IsValid()
             && "invalid ast: invalid identifier value");
      break;

    case Kind::kContextArg:
      assert(node->is_context_var() && "invalid ast: context var expected");
      val = ResolveContextVar(node->name());
      target_sp = val->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUndeclaredIdentifier,
            llvm::formatv("use of undeclared identifier '{0}'", node->name()),
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      if (!node->result_type_deref().CompareTypes(val->GetCompilerType())) {
        SetError(ErrorCode::kInvalidOperandType,
                 llvm::formatv("unexpected type of context variable '{0}' "
                               "(expected {1}, got {2})",
                               node->name(),
                               node->result_type_deref().TypeDescription(),
                               val->GetCompilerType().TypeDescription()),
                 node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      break;

    case Kind::kMemberPath:
      target_sp = m_scope->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUnknown,
            llvm::formatv(
                "unable to resolve '{0}', evaluation requires a value context",
                node->name()),
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      val = EvaluateMemberOf(m_scope, identifier.path(), false, false);
      break;

    case Kind::kThisKeyword:
      target_sp = m_scope->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUnknown,
            "unable to resolve 'this', evaluation requires a value context",
            node->location());
        m_result = lldb::ValueObjectSP();
        return;
      }
      val = m_scope->AddressOf(error);
      break;

    default:
      assert(false && "invalid ast: invalid identifier kind");
  }

  target_sp = val->GetTargetSP();
  assert(target_sp && target_sp->IsValid() &&
         "identifier doesn't resolve to a valid value");

  m_result = val;
}

void DILInterpreter::Visit(const SizeOfNode* node) {
  auto operand = node->operand();

  uint64_t deref_byte_size = 0;
  uint64_t other_byte_size = 0;
  if (auto temp = operand.GetNonReferenceType().GetByteSize(m_target.get()))
    deref_byte_size = temp.value();
  if (auto temp = operand.GetByteSize(m_target.get()))
    other_byte_size = temp.value();
  // For reference type (int&) we need to look at the referenced type.
  size_t size = operand.IsReferenceType()
                ? deref_byte_size
                : other_byte_size;
  CompilerType type = node->result_type_deref();
  m_result = ValueObject::CreateValueObjectFromBytes(m_target, &size, type);
}

void DILInterpreter::Visit(const BuiltinFunctionCallNode* node) {
  if (node->name() == "__log2") {
    assert(node->arguments().size() == 1 &&
           "invalid ast: expected exactly one argument to `__log2`");
    // Get the first (and the only) argument and evaluate it.
    auto& arg = node->arguments()[0];
    lldb::ValueObjectSP val = DILEvalNode(arg.get());
    if (!val) {
      return;
    }
    assert(val->GetCompilerType().IsInteger() &&
           "invalid ast: argument to __log2 must be an interger");

    // Use Log2_32 to match the behaviour of Visual Studio debugger.
       //uint32_t ret = llvm::Log2_32(static_cast<uint32_t>(GetUInt64(val)));
    uint32_t ret =
        llvm::Log2_32(static_cast<uint32_t>(val->GetValueAsUnsigned(0)));
    m_result = ValueObject::CreateValueObjectFromBytes(
        m_target, &ret, lldb::eBasicTypeUnsignedInt);
    return;
  }

  if (node->name() == "__findnonnull") {
    assert(node->arguments().size() == 2 &&
           "invalid ast: expected exactly two arguments to `__findnonnull`");

    auto& arg1 = node->arguments()[0];
    lldb::ValueObjectSP val1_sp = DILEvalNode(arg1.get());
    if (!val1_sp) {
      return;
    }

    lldb::ValueObjectSP val1(DILGetSPWithLock(val1_sp));
    // Resolve data address for the first argument.
    uint64_t addr;

    if (val1->GetCompilerType().IsPointerType()) {
      addr = val1->GetValueAsUnsigned(0);
    } else if (val1->GetCompilerType().IsArrayType()) {
      addr = val1->GetLoadAddress();
    } else {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv("no known conversion from '{0}' to 'T*' for 1st "
                             "argument of __findnonnull()",
                             val1->GetCompilerType().GetTypeName()),
               arg1->location());
      return;
    }

    auto& arg2 = node->arguments()[1];
    lldb::ValueObjectSP val2 = DILEvalNode(arg2.get());
    if (!val2) {
      return;
    }
    lldb::ValueObjectSP val2_sp(DILGetSPWithLock(val2));
    int64_t size = val2_sp->GetValueAsSigned(0);

    if (size < 0 || size > 100000000) {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv(
                   "passing in a buffer size ('{0}') that is negative or in "
                   "excess of 100 million to __findnonnull() is not allowed.",
                   size),
               arg2->location());
      return;
    }

    lldb::ProcessSP process = m_target->GetProcessSP();
    size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize();

    uint64_t memory = 0;
    Status error;

    for (int i = 0; i < size; ++i) {
      size_t read =
          process->ReadMemory(addr + i * ptr_size, &memory, ptr_size, error);

      if (error.Fail() || read != ptr_size) {
        SetError(ErrorCode::kUnknown,
                 llvm::formatv("error calling __findnonnull(): {0}",
                               error.AsCString() ? error.AsCString()
                                                 : "cannot read memory"),
                 node->location());
        return;
      }

      if (memory != 0) {
        m_result = ValueObject::CreateValueObjectFromBytes(
            m_target, &i, lldb::eBasicTypeInt);
        return;
      }
    }

    int ret = -1;
    m_result = ValueObject::CreateValueObjectFromBytes(
        m_target, &ret, lldb::eBasicTypeInt);
    return;
  }

  assert(false && "invalid ast: unknown builtin function");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CStyleCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  switch (node->kind()) {
    case CStyleCastKind::kArithmetic: {
      assert((type.GetCanonicalType().GetBasicTypeEnumeration() !=
              lldb::eBasicTypeInvalid) &&
             "invalid ast: target type should be a basic type.");
      // Pick an appropriate cast.
      if (rhs->GetCompilerType().IsPointerType()
          || rhs->GetCompilerType().IsNullPtrType()) {
        m_result = rhs->CastPointerToBasicType(type);
      } else if (rhs->GetCompilerType().IsScalarType()) {
        m_result = rhs->CastScalarToBasicType(type, m_error);
      } else if (rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastEnumToBasicType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to arithmetic type");
      }
      return;
    }
    case CStyleCastKind::kEnumeration: {
      assert(type.IsEnumerationType() &&
             "invalid ast: target type should be an enumeration.");

      if (rhs->GetCompilerType().IsFloat()) {
        m_result = rhs->CastFloatToEnumType(type, m_error);
      } else if (rhs->GetCompilerType().IsInteger() ||
                 rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastIntegerOrEnumToEnumType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }
    case CStyleCastKind::kPointer: {
      assert(type.IsPointerType() &&
             "invalid ast: target type should be a pointer.");
      uint64_t addr = rhs->GetCompilerType().IsArrayType()
                          ? rhs->GetLoadAddress()
                          : rhs->GetValueAsUnsigned(0);
      //: GetUInt64(rhs);
      m_result = ValueObject::CreateValueObjectFromPointer(m_target, addr, type);
      return;
    }
    case CStyleCastKind::kNullptr: {
      assert(
          (type.GetCanonicalType().GetBasicTypeEnumeration() ==
           lldb::eBasicTypeNullPtr)
          && "invalid ast: target type should be a nullptr_t.");
      m_result = ValueObject::CreateValueObjectFromNullptr(m_target, type);
      return;
    }
    case CStyleCastKind::kReference: {
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result =
          lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
      return;
    }
  }

  assert(false && "invalid ast: unexpected c-style cast kind");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CxxStaticCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  switch (node->kind()) {
    case CxxStaticCastKind::kNoOp: {
      assert(type.CompareTypes(rhs->GetCompilerType()) &&
             "invalid ast: types should be the same");
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
      return;
    }

    case CxxStaticCastKind::kArithmetic: {
      assert(type.IsScalarType());
      if (rhs->GetCompilerType().IsPointerType()
          || rhs->GetCompilerType().IsNullPtrType()) {
        assert(type.IsBoolean() && "invalid ast: target type should be bool");
        m_result = rhs->CastPointerToBasicType(type);
      } else if (rhs->GetCompilerType().IsScalarType()) {
        m_result = rhs->CastScalarToBasicType(type, m_error);
      } else if (rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastEnumToBasicType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to arithmetic type");
      }
      return;
    }

    case CxxStaticCastKind::kEnumeration: {
      if (rhs->GetCompilerType().IsFloat()) {
        m_result = rhs->CastFloatToEnumType(type, m_error);
      } else if (rhs->GetCompilerType().IsInteger() ||
                 rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastIntegerOrEnumToEnumType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }

    case CxxStaticCastKind::kPointer: {
      assert(type.IsPointerType() &&
             "invalid ast: target type should be a pointer.");

      uint64_t addr = rhs->GetCompilerType().IsArrayType()
                          ? rhs-> GetLoadAddress()
                          : rhs->GetValueAsUnsigned(0);
      //: GetUInt64(rhs);
      m_result = ValueObject::CreateValueObjectFromPointer(m_target, addr, type);
      return;
    }

    case CxxStaticCastKind::kNullptr: {
      m_result = ValueObject::CreateValueObjectFromNullptr(m_target, type);
      return;
    }

    case CxxStaticCastKind::kDerivedToBase: {
      m_result = rhs->CastDerivedToBaseType(type, node->idx());
      return;
    }

    case CxxStaticCastKind::kBaseToDerived: {
      m_result = rhs->CastBaseToDerivedType(type, node->offset());
      return;
    }
  }
}

void DILInterpreter::Visit(const CxxReinterpretCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  if (type.IsInteger()) {
    if (rhs->GetCompilerType().IsPointerType()
        || rhs->GetCompilerType().IsNullPtrType()) {
      m_result = rhs->CastPointerToBasicType(type);
    } else {
      assert(type.CompareTypes(rhs->GetCompilerType()) &&
             "invalid ast: operands should have the same type");
      // Cast value to handle type aliases.
      lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
    }
  } else if (type.IsEnumerationType()) {
    assert(type.CompareTypes(rhs->GetCompilerType()) &&
           "invalid ast: operands should have the same type");
    // Cast value to handle type aliases.
    lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
    m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
  } else if (type.IsPointerType()) {
    assert((rhs->GetCompilerType().IsInteger() ||
            rhs->GetCompilerType().IsEnumerationType() ||
            rhs->GetCompilerType().IsPointerType() ||
            rhs->GetCompilerType().IsArrayType()) &&
           "invalid ast: unexpected operand to reinterpret_cast");
    uint64_t addr = rhs->GetCompilerType().IsArrayType()
                    ? rhs->GetLoadAddress()
                    : rhs->GetValueAsUnsigned(0);
    //: GetUInt64(rhs);
    m_result = ValueObject::CreateValueObjectFromPointer(m_target, addr, type);
  } else if (type.IsReferenceType()) {
    lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
    m_result =
        lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
  } else {
    assert(false && "invalid ast: unexpected reinterpret_cast kind");
    m_result = lldb::ValueObjectSP();
  }
}

void DILInterpreter::Visit(const MemberOfNode* node) {
  assert(!node->member_index().empty() && "invalid ast: member index is empty");

  // TODO: Implement address-of elision for member-of:
  //
  //  &(*ptr).foo -> (ptr + foo_offset)
  //  &ptr->foo -> (ptr + foo_offset)
  //
  // This requires calculating the offset of "foo" and generally possible only
  // for members from non-virtual bases.

  Status error;
  lldb::ValueObjectSP lhs = DILEvalNode(node->lhs());
  if (!lhs) {
    return;
  }

  if (lhs->GetCompilerType().IsReferenceType())
    lhs = lhs->Dereference(error);
  m_result = EvaluateMemberOf(lhs, node->member_index(), node->is_synthetic(),
                              node->is_dynamic());
}

void DILInterpreter::Visit(const ArraySubscriptNode* node) {
  auto base = DILEvalNode(node->base());
  if (!base) {
    return;
  }
  auto index = DILEvalNode(node->index());
  if (!index) {
    return;
  }

  // Check to see if either the base or the index are references; if they
  // are, dereference them.
  Status error;
  if (base->GetCompilerType().IsReferenceType())
    base = base->Dereference(error);
  if (index->GetCompilerType().IsReferenceType())
    index = index->Dereference(error);

  // Check to see if 'base' has a synthetic value; if so, try using that.
  if (base->HasSyntheticValue()) {
    lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
    if (synthetic && synthetic != base) {
      //uint64_t child_idx = GetUInt64(index);
      uint64_t child_idx = index->GetValueAsUnsigned(0);
      if (static_cast<uint32_t>(child_idx) < synthetic->GetNumChildren()) {
        lldb::ValueObjectSP child_valobj_sp =
            synthetic->GetChildAtIndex(child_idx);
        if (child_valobj_sp) {
          m_result = child_valobj_sp;
          return;
        }
      }
    }
  }

  // Verify that the 'index' is not out-of-range for the declared type.
  lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
  lldb::VariableSP base_var = base->GetVariable();
  if (synthetic) {
    uint32_t num_children = synthetic->GetNumChildren();
    if (index->GetValueAsSigned(0) >= num_children) {
      SetError(ErrorCode::kSubscriptOutOfRange,
               llvm::formatv("array index {0} is not valid for \"({1}) {2}\"",
                             index->GetValueAsSigned(0),
                             base->GetTypeName().AsCString("<invalid type>"),
                             base->GetName().AsCString()),
               node->location());
      m_result = lldb::ValueObjectSP();
      return;
    }
  }

  std::string base_name(base->GetCompilerType().GetTypeName(false).AsCString());
  assert((base->GetCompilerType().IsPointerType() ||
          base_name.find("std::tuple") != std::string::npos)
         && "array subscript: base must be a pointer or a tuple");
  assert(index->GetCompilerType().IsIntegerOrUnscopedEnumerationType() &&
         "array subscript: index must be integer or unscoped enum");

  CompilerType item_type = base->GetCompilerType().GetPointeeType();
  //lldb::addr_t base_addr = GetUInt64(base);
  lldb::addr_t base_addr = base->GetValueAsUnsigned(0);

  // Create a pointer and add the index, i.e. "base + index".
  lldb::ValueObjectSP value =
      PointerAdd(ValueObject::CreateValueObjectFromPointer(
          //m_target, base_addr, item_type.GetPointerType()), GetUInt64(index));
          m_target, base_addr, item_type.GetPointerType()),
                 //index->GetValueAsUnsigned(0));
                 index->GetValueAsSigned(0));

  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    m_result = value_sp;
  } else {
    Status error;
    m_result = value_sp->Dereference(error);
  }
}

void DILInterpreter::Visit(const BinaryOpNode* node) {
  // Short-circuit logical operators.
  if (node->kind() == BinaryOpKind::LAnd || node->kind() == BinaryOpKind::LOr) {
    Status error;
    auto lhs = DILEvalNode(node->lhs());
    if (!lhs) {
      return;
    }
    if (lhs->GetCompilerType().IsReferenceType())
      lhs = lhs->Dereference(error);
    assert(lhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
           "invalid ast: must be convertible to bool");

    // For "&&" break if LHS is "false", for "||" if LHS is "true".
    bool lhs_val = lhs->GetValueAsBool();
    bool break_early =
        (node->kind() == BinaryOpKind::LAnd) ? !lhs_val : lhs_val;

    if (break_early) {
      m_result = ValueObject::CreateValueObjectFromBool(m_target, lhs_val);
      return;
    }

    // Breaking early didn't happen, evaluate the RHS and use it as a result.
    auto rhs = DILEvalNode(node->rhs());
    if (!rhs) {
      return;
    }
    if (rhs->GetCompilerType().IsReferenceType())
      rhs = rhs->Dereference(error);
    assert(rhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
           "invalid ast: must be convertible to bool");

    m_result = ValueObject::CreateValueObjectFromBool(m_target,
                                                      rhs->GetValueAsBool());
    return;
  }

  // All other binary operations require evaluating both operands.
  auto lhs = DILEvalNode(node->lhs());
  if (!lhs) {
    return;
  }
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  // For math operations, be sure to dereference the operands.
  if ((node->kind() == BinaryOpKind::Add)
      || (node->kind() == BinaryOpKind::Sub)
      || (node->kind() == BinaryOpKind::Mul)
      || (node->kind() == BinaryOpKind::Div)
      || (node->kind() == BinaryOpKind::Rem)) {
    Status error;
    if (lhs->GetCompilerType().IsReferenceType())
      lhs = lhs->Dereference(error);
    if (rhs->GetCompilerType().IsReferenceType())
      rhs = rhs->Dereference(error);
  }

  switch (node->kind()) {
    case BinaryOpKind::Add:
      m_result = EvaluateBinaryAddition(lhs, rhs);
      return;
    case BinaryOpKind::Sub:
      // The result type of subtraction is required because it holds the
      // correct "ptrdiff_t" type in the case of subtracting two pointers.
      m_result = EvaluateBinarySubtraction(lhs, rhs, node->result_type_deref());
      return;
    case BinaryOpKind::Mul:
      m_result = EvaluateBinaryMultiplication(lhs, rhs);
      return;
    case BinaryOpKind::Div:
      m_result = EvaluateBinaryDivision(lhs, rhs);
      return;
    case BinaryOpKind::Rem:
      m_result = EvaluateBinaryRemainder(lhs, rhs);
      return;
    case BinaryOpKind::And:
    case BinaryOpKind::Or:
    case BinaryOpKind::Xor:
      m_result = EvaluateBinaryBitwise(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::Shl:
    case BinaryOpKind::Shr:
      m_result = EvaluateBinaryShift(node->kind(), lhs, rhs);
      return;

    // Comparison operations.
    case BinaryOpKind::EQ:
    case BinaryOpKind::NE:
    case BinaryOpKind::LT:
    case BinaryOpKind::LE:
    case BinaryOpKind::GT:
    case BinaryOpKind::GE:
      m_result = EvaluateComparison(node->kind(), lhs, rhs);
      return;

    case BinaryOpKind::Assign:
      m_result = EvaluateAssignment(lhs, rhs);
      return;

    case BinaryOpKind::AddAssign:
      m_result = EvaluateBinaryAddAssign(lhs, rhs);
      return;
    case BinaryOpKind::SubAssign:
      m_result = EvaluateBinarySubAssign(lhs, rhs);
      return;
    case BinaryOpKind::MulAssign:
      m_result = EvaluateBinaryMulAssign(lhs, rhs);
      return;
    case BinaryOpKind::DivAssign:
      m_result = EvaluateBinaryDivAssign(lhs, rhs);
      return;
    case BinaryOpKind::RemAssign:
      m_result = EvaluateBinaryRemAssign(lhs, rhs);
      return;

    case BinaryOpKind::AndAssign:
    case BinaryOpKind::OrAssign:
    case BinaryOpKind::XorAssign:
      m_result = EvaluateBinaryBitwiseAssign(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::ShlAssign:
    case BinaryOpKind::ShrAssign:
      m_result = EvaluateBinaryShiftAssign(node->kind(), lhs, rhs,
                                          node->comp_assign_type());
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const UnaryOpNode* node) {
  FlowAnalysis rhs_flow(
      /* address_of_is_pending */ node->kind() == UnaryOpKind::AddrOf);

  Status error;
  auto rhs = DILEvalNode(node->rhs(), &rhs_flow);
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType())
    rhs = rhs->Dereference(error);

  switch (node->kind()) {
    case UnaryOpKind::Deref:
      //if (rhs->GetVariable())
      {
        lldb::ValueObjectSP dynamic_rhs =
            rhs->GetDynamicValue(m_default_dynamic);
        if (dynamic_rhs)
          rhs = dynamic_rhs;
      }
      m_result = EvaluateDereference(rhs);
      return;
    case UnaryOpKind::AddrOf:
      // If the address-of operation wasn't cancelled during the evaluation of
      // RHS (e.g. because of the address-of-a-dereference elision), apply it
      // here.
      if (rhs_flow.AddressOfIsPending()) {
        Status error;
        lldb::ValueObjectSP rhs_sp(DILGetSPWithLock(rhs));
        m_result = rhs_sp->AddressOf(error);
      } else {
        m_result = rhs;
      }
      return;
    case UnaryOpKind::Plus:
      m_result = rhs;
      return;
    case UnaryOpKind::Minus:
      m_result = EvaluateUnaryMinus(rhs);
      return;
    case UnaryOpKind::LNot:
      m_result = EvaluateUnaryNegation(rhs);
      return;
    case UnaryOpKind::Not:
      m_result = EvaluateUnaryBitwiseNot(rhs);
      return;
    case UnaryOpKind::PreInc:
      m_result = EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PreDec:
      m_result = EvaluateUnaryPrefixDecrement(rhs);
      return;
    case UnaryOpKind::PostInc:
      // In postfix inc/dec the result is the original value.
      m_result = rhs->Clone(ConstString("cloned-object"));
      EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PostDec:
      // In postfix inc/dec the result is the original value.
      m_result = rhs->Clone(ConstString("cloned-object"));
      EvaluateUnaryPrefixDecrement(rhs);
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const TernaryOpNode* node) {
  auto cond = DILEvalNode(node->cond());
  if (!cond) {
    return;
  }
  assert(cond->GetCompilerType().IsContextuallyConvertibleToBool() &&
         "invalid ast: must be convertible to bool");

  // Pass down the flow analysis because the conditional operator is a "flow
  // control" construct -- LHS/RHS might be lvalues and eligible for some
  // optimizations (e.g. "&*" elision).
  if (cond->GetValueAsBool()) {
    m_result = DILEvalNode(node->lhs(), flow_analysis());
  } else {
    m_result = DILEvalNode(node->rhs(), flow_analysis());
  }
}

void DILInterpreter::Visit(const SmartPtrToPtrDecay* node) {
  auto ptr = DILEvalNode(node->ptr());
  if (!ptr) {
    return;
  }

  assert(IsSmartPtrType(ptr->GetCompilerType()) &&
         "invalid ast: must be a smart pointer");

  // Prefer synthetic value because we need LLDB machinery to "dereference" the
  // pointer for us. This is usually the default, but if the value was obtained
  // as a field of some other object, it will inherit the value from parent.
  lldb::ValueObjectSP ptr_value = ptr;
  bool prefer_synthetic_value = true;
  lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues;
  lldb::TargetSP target_sp  = ptr_value->GetTargetSP();
  if (target_sp)
    use_dynamic = target_sp->GetPreferDynamicValue();
  lldb::ValueObjectSP value_sp(DILGetSPWithLock(ptr_value, use_dynamic,
                                                prefer_synthetic_value));
  ptr_value = value_sp->GetChildAtIndex(0);

  lldb::addr_t base_addr = ptr_value->GetValueAsUnsigned(0);
  CompilerType pointer_type = ptr_value->GetCompilerType();

  if (value_sp->GetCompilerType().IsTemplateType()
      && value_sp->GetCompilerType().GetNumTemplateArguments() == 1)
    pointer_type =
        value_sp->GetCompilerType().GetTypeTemplateArgument(0).GetPointerType();

  m_result = ValueObject::CreateValueObjectFromPointer(m_target, base_addr,
                                                       pointer_type);

  ValueObject *deref_valobj = nullptr;
  if (ptr->HasSyntheticValue())
    deref_valobj =
        ptr->GetSyntheticValue()->GetChildMemberWithName("$$dereference$$").get();
  else if (ptr->IsSynthetic())
    deref_valobj = ptr->GetChildMemberWithName("$$dereference$$").get();
  if (deref_valobj)
    //m_result->m_deref_valobj = deref_valobj;
    m_result->SetDerefValobj(deref_valobj);
}

lldb::ValueObjectSP DILInterpreter::EvaluateComparison(BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  // Evaluate arithmetic operation for two integral values.
  if (lhs->GetCompilerType().IsInteger() && rhs->GetCompilerType().IsInteger()) {
    bool ret = Compare(kind, lhs->GetValueAsAPSInt(), rhs->GetValueAsAPSInt());
    return ValueObject::CreateValueObjectFromBool(m_target, ret);
  }

  // Evaluate arithmetic operation for two floating point values.
  if (lhs->GetCompilerType().IsFloat() && rhs->GetCompilerType().IsFloat()) {
    bool ret = Compare(kind, lhs->GetValueAsFloat(),
                       rhs->GetValueAsFloat());
    return ValueObject::CreateValueObjectFromBool(m_target, ret);
  }

  // Evaluate arithmetic operation for two scoped enum values.
  if (lhs->GetCompilerType().IsScopedEnumerationType()
      && rhs->GetCompilerType().IsScopedEnumerationType()) {
    bool ret = Compare(kind, lhs->GetValueAsAPSInt(), rhs->GetValueAsAPSInt());
    return ValueObject::CreateValueObjectFromBool(m_target, ret);
  }

  // Must be pointer/integer and/or nullptr comparison.
  size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize() * 8;

  bool ret =
      Compare(kind, llvm::APSInt(lhs->GetValueAsAPSInt().sextOrTrunc(ptr_size),
                                 true),
              llvm::APSInt(rhs->GetValueAsAPSInt().sextOrTrunc(ptr_size),
                           true));
  return ValueObject::CreateValueObjectFromBool(m_target, ret);
}

lldb::ValueObjectSP DILInterpreter::EvaluateDereference(lldb::ValueObjectSP rhs)
{
  // If rhs is a reference, dereference it first.
  Status error;
  if (rhs->GetCompilerType().IsReferenceType())
    rhs = rhs->Dereference(error);

  assert(rhs->GetCompilerType().IsPointerType()
         && "invalid ast: must be a pointer type");

  if (rhs->GetDerefValobj())
    return rhs->GetDerefValobj()->GetSP();

  CompilerType pointer_type = rhs->GetCompilerType();
  //lldb::addr_t base_addr = GetUInt64(rhs);
  lldb::addr_t base_addr = rhs->GetValueAsUnsigned(0);

  lldb::ValueObjectSP value = ValueObject::CreateValueObjectFromPointer(
      m_target, base_addr,  pointer_type);

  lldb::ValueObjectSP value_sp(DILGetSPWithLock(value));
  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    return value_sp;
  }

  return value_sp->Dereference(error);
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryMinus(lldb::ValueObjectSP rhs)
{
  assert((rhs->GetCompilerType().IsInteger() || rhs->GetCompilerType().IsFloat())
         && "invalid ast: must be an arithmetic type");

  if (rhs->GetCompilerType().IsInteger()) {
    llvm::APSInt v = rhs->GetValueAsAPSInt();
    v.negate();
    return ValueObject::CreateValueObjectFromAPInt(m_target, v,
                                                   rhs->GetCompilerType());
  }
  if (rhs->GetCompilerType().IsFloat()) {
    llvm::APFloat v = rhs->GetValueAsFloat();
    v.changeSign();
    return ValueObject::CreateValueObjectFromAPFloat(m_target, v,
                                                     rhs->GetCompilerType());
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryNegation(
    lldb::ValueObjectSP rhs)
{
  assert(rhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
         "invalid ast: must be convertible to bool");
  return ValueObject::CreateValueObjectFromBool(m_target,
                                                !rhs->GetValueAsBool());
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryBitwiseNot(
    lldb::ValueObjectSP rhs) {
  assert(rhs->GetCompilerType().IsInteger() && "invalid ast: must be an integer");
  llvm::APSInt v = rhs->GetValueAsAPSInt();
  v.flipAllBits();
  return ValueObject::CreateValueObjectFromAPInt(m_target, v,
                                                 rhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixIncrement(
    lldb::ValueObjectSP rhs)
{
  assert((rhs->GetCompilerType().IsInteger() || rhs->GetCompilerType().IsFloat()
          || rhs->GetCompilerType().IsPointerType()) &&
         "invalid ast: must be either arithmetic type or pointer");

  if (rhs->GetCompilerType().IsInteger()) {
    llvm::APSInt v = rhs->GetValueAsAPSInt();
    ++v;  // Do the increment.

    rhs->UpdateIntegerValue(v);
    return rhs;
  }
  if (rhs->GetCompilerType().IsFloat()) {
    llvm::APFloat v = rhs->GetValueAsFloat();
    // Do the increment.
    v = v + llvm::APFloat(v.getSemantics(), 1ULL);

    rhs->UpdateIntegerValue(v.bitcastToAPInt());
    return rhs;
  }
  if (rhs->GetCompilerType().IsPointerType()) {
    //uint64_t v = GetUInt64(rhs);
    uint64_t v = rhs->GetValueAsUnsigned(0);
    uint64_t byte_size = 0;
    if (auto temp =
            rhs->GetCompilerType().GetPointeeType().GetByteSize(
                rhs->GetTargetSP().get()))
      byte_size = temp.value();
    v += byte_size;;  // Do the increment.

    rhs->UpdateIntegerValue(llvm::APInt(64, v));
    return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixDecrement(
    lldb::ValueObjectSP rhs) {
  assert((rhs->GetCompilerType().IsInteger() ||
          rhs->GetCompilerType().IsFloat() ||
          rhs->GetCompilerType().IsPointerType()) &&
         "invalid ast: must be either arithmetic type or pointer");

  if (rhs->GetCompilerType().IsInteger()) {
    llvm::APSInt v = rhs->GetValueAsAPSInt();
    --v;  // Do the decrement.

    rhs->UpdateIntegerValue(v);
    return rhs;
  }
  if (rhs->GetCompilerType().IsFloat()) {
    llvm::APFloat v = rhs->GetValueAsFloat();
    // Do the decrement.
    v = v - llvm::APFloat(v.getSemantics(), 1ULL);

    rhs->UpdateIntegerValue(v.bitcastToAPInt());
    return rhs;
  }
  if (rhs->GetCompilerType().IsPointerType()) {
    //uint64_t v = GetUInt64(rhs);
    uint64_t v = rhs->GetValueAsUnsigned(0);
    uint64_t byte_size = 0;
    if (auto temp =
            rhs->GetCompilerType().GetPointeeType().GetByteSize(
                rhs->GetTargetSP().get()))
      byte_size = temp.value();
    v -= byte_size;  // Do the decrement.

    rhs->UpdateIntegerValue(llvm::APInt(64, v));
    return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddition(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  // Addition of two arithmetic types.
  if (lhs->GetCompilerType().IsScalarType()
      && rhs->GetCompilerType().IsScalarType()) {
    assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Add, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }

  // Here one of the operands must be a pointer and the other one an integer.
  lldb::ValueObjectSP ptr, offset;
  if (lhs->GetCompilerType().IsPointerType()) {
    ptr = lhs;
    offset = rhs;
  } else {
    ptr = rhs;
    offset = lhs;
  }
  assert(ptr->GetCompilerType().IsPointerType() &&
         "invalid ast: ptr must be a pointer");
  assert(offset->GetCompilerType().IsInteger() &&
         "invalid ast: offset must be an integer");

  //if (GetUInt64(ptr) == 0 && GetUInt64(offset) != 0) {
  if (ptr->GetValueAsUnsigned(0) == 0 && offset->GetValueAsUnsigned(0) != 0) {
    // Binary addition with null pointer causes mismatches between LLDB and
    // lldb-eval if the offset different than zero.
    SetUbStatus(m_error, ErrorCode::kUBNullPtrArithmetic);
  }

  //return PointerAdd(ptr, GetUInt64(offset));
  return PointerAdd(ptr, offset->GetValueAsUnsigned(0));
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubtraction(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs,
    CompilerType result_type)
{
  if (lhs->GetCompilerType().IsScalarType()
      && rhs->GetCompilerType().IsScalarType()) {
    assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Sub, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }
  assert(lhs->GetCompilerType().IsPointerType()
         && "invalid ast: lhs must be a pointer");

  // "pointer - integer" operation.
  if (rhs->GetCompilerType().IsInteger()) {
    //return PointerAdd(lhs, -GetUInt64(rhs));
    return PointerAdd(lhs, - rhs->GetValueAsUnsigned(0));
  }

  uint64_t lhs_byte_size = 0;
  uint64_t rhs_byte_size = 0;
  if (auto temp =
      lhs->GetCompilerType().GetPointeeType().GetByteSize(
          lhs->GetTargetSP().get()))
    lhs_byte_size = temp.value();
  if (auto temp = rhs->GetCompilerType().GetPointeeType().GetByteSize(
          rhs->GetTargetSP().get()))
    rhs_byte_size = temp.value();
  // "pointer - pointer" operation.
  assert(rhs->GetCompilerType().IsPointerType()
         && "invalid ast: rhs must an integer or a pointer");
  assert((lhs_byte_size == rhs_byte_size) &&
         "invalid ast: pointees should be the same size");

  // Since pointers have compatible types, both have the same pointee size.
  uint64_t item_size = lhs_byte_size;
  // Pointer difference is a signed value.
  //int64_t diff = static_cast<int64_t>(GetUInt64(lhs) - GetUInt64(rhs));
  int64_t diff = static_cast<int64_t>(lhs->GetValueAsUnsigned(0) -
                                      rhs->GetValueAsUnsigned(0));

  if (diff % item_size != 0 && diff < 0) {
    // If address difference isn't divisible by pointee size then performing
    // the operation is undefined behaviour. Note: mismatches were encountered
    // only for negative difference (diff < 0).
    SetUbStatus(m_error, ErrorCode::kUBInvalidPtrDiff);
  }

  diff /= static_cast<int64_t>(item_size);

  // Pointer difference is ptrdiff_t.
  return ValueObject::CreateValueObjectFromBytes(m_target, &diff, result_type);
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMultiplication(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((lhs->GetCompilerType().IsScalarType() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Mul, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivision(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((lhs->GetCompilerType().IsScalarType() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  // Check for zero only for integer division.
  //if (rhs->GetCompilerType().IsInteger() && GetUInt64(rhs) == 0) {
  if (rhs->GetCompilerType().IsInteger() && rhs->GetValueAsUnsigned(0) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: division by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (rhs->GetCompilerType().IsInteger() && IsInvalidDivisionByMinusOne(lhs, rhs))
  {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Div, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemainder(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  assert((lhs->GetCompilerType().IsInteger()
          && lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()))
         && "invalid ast: operands must be integers and have the same type");

  //if (GetUInt64(rhs) == 0) {
  if (rhs->GetValueAsUnsigned(0) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: remainder by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (IsInvalidDivisionByMinusOne(lhs, rhs)) {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOpInteger(m_target, BinaryOpKind::Rem, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwise(
    BinaryOpKind kind,
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  assert((lhs->GetCompilerType().IsInteger()
          && lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()))
         &&"invalid ast: operands must be integers and have the same type");
  assert((kind == BinaryOpKind::And || kind == BinaryOpKind::Or ||
          kind == BinaryOpKind::Xor) &&
         "invalid ast: operation must be '&', '|' or '^'");

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShift(BinaryOpKind kind,
                                                        lldb::ValueObjectSP lhs,
                                                        lldb::ValueObjectSP rhs)
{
  assert(lhs->GetCompilerType().IsInteger() &&
         rhs->GetCompilerType().IsInteger() &&
         "invalid ast: operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr) &&
         "invalid ast: operation must be '<<' or '>>'");

  uint64_t lhs_byte_size = 0;
  if (auto temp = lhs->GetCompilerType().GetByteSize(lhs->GetTargetSP().get()))
    lhs_byte_size = temp.value();
  // Performing shift operation is undefined behaviour if the right operand
  // isn't in interval [0, bit-size of the left operand).
  if (rhs->GetValueAsAPSInt().isNegative() ||
      (rhs->GetValueAsUnsigned(0) >= lhs_byte_size * CHAR_BIT)) {
    //(GetUInt64(rhs) >= lhs_byte_size * CHAR_BIT)) {
    SetUbStatus(m_error, ErrorCode::kUBInvalidShift);
  }

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateAssignment(lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
         "invalid ast: operands must have the same type");

  lhs->UpdateIntegerValue(rhs);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs) {
  lldb::ValueObjectSP ret;

  if (lhs->GetCompilerType().IsPointerType()) {
    assert(rhs->GetCompilerType().IsInteger() &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinaryAddition(lhs, rhs);
  } else {
    assert(lhs->GetCompilerType().IsScalarType()
           && "invalid ast: lhs must be an arithmetic type");
    assert(rhs->GetCompilerType().IsBasicType() &&
           "invalid ast: rhs must be a basic type");
    ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(), m_error);
    ret = EvaluateBinaryAddition(ret, rhs);
    ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);
  }

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  lldb::ValueObjectSP ret;

  if (lhs->GetCompilerType().IsPointerType()) {
    assert(rhs->GetCompilerType().IsInteger() &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinarySubtraction(lhs, rhs, lhs->GetCompilerType());
  } else {
    assert(lhs->GetCompilerType().IsScalarType()
           && "invalid ast: lhs must be an arithmetic type");
    assert(rhs->GetCompilerType().IsBasicType() &&
           "invalid ast: rhs must be a basic type");
    ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(), m_error);
    ret = EvaluateBinarySubtraction(ret, rhs, ret->GetCompilerType());
    ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);
  }

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMulAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(),
                                                       m_error);
  ret = EvaluateBinaryMultiplication(ret, rhs);
  ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(),
                                                       m_error);
  ret = EvaluateBinaryDivision(ret, rhs);
  ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(),
                                                       m_error);
  ret = EvaluateBinaryRemainder(ret, rhs);
  ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwiseAssign(
    BinaryOpKind kind, lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs)
{
  switch (kind) {
    case BinaryOpKind::AndAssign:
      kind = BinaryOpKind::And;
      break;
    case BinaryOpKind::OrAssign:
      kind = BinaryOpKind::Or;
      break;
    case BinaryOpKind::XorAssign:
      kind = BinaryOpKind::Xor;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '&=', '|=' or '^='");
      break;
  }
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType() &&
         "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastScalarToBasicType(rhs->GetCompilerType(),
                                                       m_error);
  ret = EvaluateBinaryBitwise(kind, ret, rhs);
  ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShiftAssign(
    BinaryOpKind kind,
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs,
    CompilerType comp_assign_type)
{
  switch (kind) {
    case BinaryOpKind::ShlAssign:
      kind = BinaryOpKind::Shl;
      break;
    case BinaryOpKind::ShrAssign:
      kind = BinaryOpKind::Shr;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '<<=' or '>>='");
      break;
  }
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType() &&
         "invalid ast: rhs must be a basic type");
  assert(comp_assign_type.IsInteger() &&
         "invalid ast: comp_assign_type must be an integer");

  lldb::ValueObjectSP ret = lhs->CastScalarToBasicType(comp_assign_type,
                                                       m_error);
  ret = EvaluateBinaryShift(kind, ret, rhs);
  ret = ret->CastScalarToBasicType(lhs->GetCompilerType(), m_error);

  lhs->UpdateIntegerValue(ret);
  return lhs;
}

lldb::ValueObjectSP DILInterpreter::PointerAdd(lldb::ValueObjectSP lhs,
                                               int64_t offset) {
  uint64_t byte_size = 0;
  if (auto temp = lhs->GetCompilerType().GetPointeeType().GetByteSize(
          lhs->GetTargetSP().get()))
    byte_size = temp.value();
  //uintptr_t addr = GetUInt64(lhs) + offset * byte_size;
  uintptr_t addr = lhs->GetValueAsUnsigned(0) + offset * byte_size;

  return ValueObject::CreateValueObjectFromPointer(m_target, addr,
                                                   lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::ResolveContextVar(
    const std::string& name) const
{
  auto it = m_context_vars.find(name);
  return it != m_context_vars.end() ? it->second : lldb::ValueObjectSP();
}

}  // namespace lldb_private
