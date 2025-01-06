//===-- FormatterBytecode.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatterBytecode.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadicDetails.h"

using namespace lldb;
namespace lldb_private {

std::string toString(FormatterBytecode::OpCodes op) {
  switch (op) {
#define DEFINE_OPCODE(OP, MNEMONIC, NAME)                                      \
  case OP: {                                                                   \
    const char *s = MNEMONIC;                                                  \
    return s ? s : #NAME;                                                      \
  }
#include "FormatterBytecode.def"
#undef DEFINE_SIGNATURE
  }
  return llvm::utostr(op);
}

std::string toString(FormatterBytecode::Selectors sel) {
  switch (sel) {
#define DEFINE_SELECTOR(ID, NAME)                                              \
  case ID:                                                                     \
    return "@" #NAME;
#include "FormatterBytecode.def"
#undef DEFINE_SIGNATURE
  }
  return "@" + llvm::utostr(sel);
}

std::string toString(FormatterBytecode::Signatures sig) {
  switch (sig) {
#define DEFINE_SIGNATURE(ID, NAME)                                             \
  case ID:                                                                     \
    return "@" #NAME;
#include "FormatterBytecode.def"
#undef DEFINE_SIGNATURE
  }
  return llvm::utostr(sig);
}

std::string toString(const FormatterBytecode::DataStack &data) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "[ ";
  for (auto &d : data) {
    if (auto s = std::get_if<std::string>(&d))
      os << '"' << *s << '"';
    else if (auto u = std::get_if<uint64_t>(&d))
      os << *u << 'u';
    else if (auto i = std::get_if<int64_t>(&d))
      os << *i;
    else if (auto valobj = std::get_if<ValueObjectSP>(&d)) {
      if (!valobj->get())
        os << "null";
      else
        os << "object(" << valobj->get()->GetValueAsCString() << ')';
    } else if (auto type = std::get_if<CompilerType>(&d)) {
      os << '(' << type->GetTypeName(true) << ')';
    } else if (auto sel = std::get_if<FormatterBytecode::Selectors>(&d)) {
      os << toString(*sel);
    }
    os << ' ';
  }
  os << ']';
  return s;
}

namespace FormatterBytecode {

/// Implement the @format function.
static llvm::Error FormatImpl(DataStack &data) {
  auto fmt = data.Pop<std::string>();
  auto replacements =
      llvm::formatv_object_base::parseFormatString(fmt, 0, false);
  std::string s;
  llvm::raw_string_ostream os(s);
  unsigned num_args = 0;
  for (const auto &r : replacements)
    if (r.Type == llvm::ReplacementType::Format)
      num_args = std::max(num_args, r.Index + 1);

  if (data.size() < num_args)
    return llvm::createStringError("not enough arguments");

  for (const auto &r : replacements) {
    if (r.Type == llvm::ReplacementType::Literal) {
      os << r.Spec;
      continue;
    }
    using namespace llvm::support::detail;
    auto arg = data[data.size() - num_args + r.Index];
    auto format = [&](format_adapter &&adapter) {
      llvm::FmtAlign Align(adapter, r.Where, r.Width, r.Pad);
      Align.format(os, r.Options);
    };

    if (auto s = std::get_if<std::string>(&arg))
      format(build_format_adapter(s->c_str()));
    else if (auto u = std::get_if<uint64_t>(&arg))
      format(build_format_adapter(u));
    else if (auto i = std::get_if<int64_t>(&arg))
      format(build_format_adapter(i));
    else if (auto valobj = std::get_if<ValueObjectSP>(&arg)) {
      if (!valobj->get())
        format(build_format_adapter("null object"));
      else
        format(build_format_adapter(valobj->get()->GetValueAsCString()));
    } else if (auto type = std::get_if<CompilerType>(&arg))
      format(build_format_adapter(type->GetDisplayTypeName()));
    else if (auto sel = std::get_if<FormatterBytecode::Selectors>(&arg))
      format(build_format_adapter(toString(*sel)));
  }
  data.Push(s);
  return llvm::Error::success();
}

static llvm::Error TypeCheck(llvm::ArrayRef<DataStackElement> data,
                             DataType type) {
  if (data.size() < 1)
    return llvm::createStringError("not enough elements on data stack");

  auto &elem = data.back();
  switch (type) {
  case Any:
    break;
  case String:
    if (!std::holds_alternative<std::string>(elem))
      return llvm::createStringError("expected String");
    break;
  case UInt:
    if (!std::holds_alternative<uint64_t>(elem))
      return llvm::createStringError("expected UInt");
    break;
  case Int:
    if (!std::holds_alternative<int64_t>(elem))
      return llvm::createStringError("expected Int");
    break;
  case Object:
    if (!std::holds_alternative<ValueObjectSP>(elem))
      return llvm::createStringError("expected Object");
    break;
  case Type:
    if (!std::holds_alternative<CompilerType>(elem))
      return llvm::createStringError("expected Type");
    break;
  case Selector:
    if (!std::holds_alternative<Selectors>(elem))
      return llvm::createStringError("expected Selector");
    break;
  }
  return llvm::Error::success();
}

static llvm::Error TypeCheck(llvm::ArrayRef<DataStackElement> data,
                             DataType type1, DataType type2) {
  if (auto error = TypeCheck(data, type2))
    return error;
  return TypeCheck(data.drop_back(), type1);
}

static llvm::Error TypeCheck(llvm::ArrayRef<DataStackElement> data,
                             DataType type1, DataType type2, DataType type3) {
  if (auto error = TypeCheck(data, type3))
    return error;
  return TypeCheck(data.drop_back(1), type2, type1);
}

llvm::Error Interpret(std::vector<ControlStackElement> &control,
                      DataStack &data, Selectors sel) {
  if (control.empty())
    return llvm::Error::success();
  // Since the only data types are single endian and ULEBs, the
  // endianness should not matter.
  llvm::DataExtractor cur_block(control.back(), true, 64);
  llvm::DataExtractor::Cursor pc(0);

  while (!control.empty()) {
    /// Activate the top most block from the control stack.
    auto activate_block = [&]() {
      // Save the return address.
      if (control.size() > 1)
        control[control.size() - 2] = cur_block.getData().drop_front(pc.tell());
      cur_block = llvm::DataExtractor(control.back(), true, 64);
      if (pc)
        pc = llvm::DataExtractor::Cursor(0);
    };

    /// Fetch the next byte in the instruction stream.
    auto next_byte = [&]() -> uint8_t {
      // At the end of the current block?
      while (pc.tell() >= cur_block.size() && !control.empty()) {
        if (control.size() == 1) {
          control.pop_back();
          return 0;
        }
        control.pop_back();
        activate_block();
      }

      // Fetch the next instruction.
      return cur_block.getU8(pc);
    };

    // Fetch the next opcode.
    OpCodes opcode = (OpCodes)next_byte();
    if (control.empty() || !pc)
      return pc.takeError();

    LLDB_LOGV(GetLog(LLDBLog::DataFormatters),
              "[eval {0}] opcode={1}, control={2}, data={3}", toString(sel),
              toString(opcode), control.size(), toString(data));

    // Various shorthands to improve the readability of error handling.
#define TYPE_CHECK(...)                                                        \
  if (auto error = TypeCheck(data, __VA_ARGS__))                               \
    return error;

    auto error = [&](llvm::Twine msg) {
      return llvm::createStringError(msg + "(opcode=" + toString(opcode) + ")");
    };

    switch (opcode) {
    // Data stack manipulation.
    case op_dup:
      TYPE_CHECK(Any);
      data.Push(data.back());
      continue;
    case op_drop:
      TYPE_CHECK(Any);
      data.pop_back();
      continue;
    case op_pick: {
      TYPE_CHECK(UInt);
      uint64_t idx = data.Pop<uint64_t>();
      if (idx >= data.size())
        return error("index out of bounds");
      data.Push(data[idx]);
      continue;
    }
    case op_over:
      TYPE_CHECK(Any, Any);
      data.Push(data[data.size() - 2]);
      continue;
    case op_swap: {
      TYPE_CHECK(Any, Any);
      auto x = data.PopAny();
      auto y = data.PopAny();
      data.Push(x);
      data.Push(y);
      continue;
    }
    case op_rot: {
      TYPE_CHECK(Any, Any, Any);
      auto z = data.PopAny();
      auto y = data.PopAny();
      auto x = data.PopAny();
      data.Push(z);
      data.Push(x);
      data.Push(y);
      continue;
    }

    // Control stack manipulation.
    case op_begin: {
      uint64_t length = cur_block.getULEB128(pc);
      if (!pc)
        return pc.takeError();
      llvm::StringRef block = cur_block.getBytes(pc, length);
      if (!pc)
        return pc.takeError();
      control.push_back(block);
      continue;
    }
    case op_if:
      TYPE_CHECK(UInt);
      if (data.Pop<uint64_t>() != 0) {
        if (!cur_block.size())
          return error("empty control stack");
        activate_block();
      } else
        control.pop_back();
      continue;
    case op_ifelse:
      TYPE_CHECK(UInt);
      if (cur_block.size() < 2)
        return error("empty control stack");
      if (data.Pop<uint64_t>() == 0)
        control[control.size() - 2] = control.back();
      control.pop_back();
      activate_block();
      continue;
    case op_return:
      control.clear();
      return pc.takeError();

    // Literals.
    case op_lit_uint:
      data.Push(cur_block.getULEB128(pc));
      continue;
    case op_lit_int:
      data.Push(cur_block.getSLEB128(pc));
      continue;
    case op_lit_selector:
      data.Push(Selectors(cur_block.getU8(pc)));
      continue;
    case op_lit_string: {
      uint64_t length = cur_block.getULEB128(pc);
      llvm::StringRef bytes = cur_block.getBytes(pc, length);
      data.Push(bytes.str());
      continue;
    }
    case op_as_uint: {
      TYPE_CHECK(Int);
      uint64_t casted;
      int64_t val = data.Pop<int64_t>();
      memcpy(&casted, &val, sizeof(val));
      data.Push(casted);
      continue;
    }
    case op_as_int: {
      TYPE_CHECK(UInt);
      int64_t casted;
      uint64_t val = data.Pop<uint64_t>();
      memcpy(&casted, &val, sizeof(val));
      data.Push(casted);
      continue;
    }
    case op_is_null: {
      TYPE_CHECK(Object);
      data.Push(data.Pop<ValueObjectSP>() ? (uint64_t)0 : (uint64_t)1);
      continue;
    }

    // Arithmetic, logic, etc.
#define BINOP_IMPL(OP, CHECK_ZERO)                                             \
  {                                                                            \
    TYPE_CHECK(Any, Any);                                                      \
    auto y = data.PopAny();                                                    \
    if (std::holds_alternative<uint64_t>(y)) {                                 \
      if (CHECK_ZERO && !std::get<uint64_t>(y))                                \
        return error(#OP " by zero");                                          \
      TYPE_CHECK(UInt);                                                        \
      data.Push((uint64_t)(data.Pop<uint64_t>() OP std::get<uint64_t>(y)));    \
    } else if (std::holds_alternative<int64_t>(y)) {                           \
      if (CHECK_ZERO && !std::get<int64_t>(y))                                 \
        return error(#OP " by zero");                                          \
      TYPE_CHECK(Int);                                                         \
      data.Push((int64_t)(data.Pop<int64_t>() OP std::get<int64_t>(y)));       \
    } else                                                                     \
      return error("unsupported data types");                                  \
  }
#define BINOP(OP) BINOP_IMPL(OP, false)
#define BINOP_CHECKZERO(OP) BINOP_IMPL(OP, true)
    case op_plus:
      BINOP(+);
      continue;
    case op_minus:
      BINOP(-);
      continue;
    case op_mul:
      BINOP(*);
      continue;
    case op_div:
      BINOP_CHECKZERO(/);
      continue;
    case op_mod:
      BINOP_CHECKZERO(%);
      continue;
    case op_shl:
#define SHIFTOP(OP, LEFT)                                                      \
  {                                                                            \
    TYPE_CHECK(Any, UInt);                                                     \
    uint64_t y = data.Pop<uint64_t>();                                         \
    if (y > 64)                                                                \
      return error("shift out of bounds");                                     \
    if (std::holds_alternative<uint64_t>(data.back())) {                       \
      uint64_t x = data.Pop<uint64_t>();                                       \
      data.Push(x OP y);                                                       \
    } else if (std::holds_alternative<int64_t>(data.back())) {                 \
      int64_t x = data.Pop<int64_t>();                                         \
      if (x < 0 && LEFT)                                                       \
        return error("left shift of negative value");                          \
      if (y > 64)                                                              \
        return error("shift out of bounds");                                   \
      data.Push(x OP y);                                                       \
    } else                                                                     \
      return error("unsupported data types");                                  \
  }
      SHIFTOP(<<, true);
      continue;
    case op_shr:
      SHIFTOP(>>, false);
      continue;
    case op_and:
      BINOP(&);
      continue;
    case op_or:
      BINOP(|);
      continue;
    case op_xor:
      BINOP(^);
      continue;
    case op_not:
      TYPE_CHECK(UInt);
      data.Push(~data.Pop<uint64_t>());
      continue;
    case op_eq:
      BINOP(==);
      continue;
    case op_neq:
      BINOP(!=);
      continue;
    case op_lt:
      BINOP(<);
      continue;
    case op_gt:
      BINOP(>);
      continue;
    case op_le:
      BINOP(<=);
      continue;
    case op_ge:
      BINOP(>=);
      continue;
    case op_call: {
      TYPE_CHECK(Selector);
      Selectors sel = data.Pop<Selectors>();

      // Shorthand to improve readability.
#define POP_VALOBJ(VALOBJ)                                                     \
  auto VALOBJ = data.Pop<ValueObjectSP>();                                     \
  if (!VALOBJ)                                                                 \
    return error("null object");

      auto sel_error = [&](const char *msg) {
        return llvm::createStringError("{0} (opcode={1}, selector={2})", msg,
                                       toString(opcode).c_str(),
                                       toString(sel).c_str());
      };

      switch (sel) {
      case sel_summary: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        const char *summary = valobj->GetSummaryAsCString();
        data.Push(summary ? std::string(valobj->GetSummaryAsCString())
                          : std::string());
        break;
      }
      case sel_get_num_children: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        auto result = valobj->GetNumChildren();
        if (!result)
          return result.takeError();
        data.Push((uint64_t)*result);
        break;
      }
      case sel_get_child_at_index: {
        TYPE_CHECK(Object, UInt);
        auto index = data.Pop<uint64_t>();
        POP_VALOBJ(valobj);
        data.Push(valobj->GetChildAtIndex(index));
        break;
      }
      case sel_get_child_with_name: {
        TYPE_CHECK(Object, String);
        auto name = data.Pop<std::string>();
        POP_VALOBJ(valobj);
        data.Push(valobj->GetChildMemberWithName(name));
        break;
      }
      case sel_get_child_index: {
        TYPE_CHECK(Object, String);
        auto name = data.Pop<std::string>();
        POP_VALOBJ(valobj);
        data.Push((uint64_t)valobj->GetIndexOfChildWithName(name));
        break;
      }
      case sel_get_type: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        // FIXME: do we need to control dynamic type resolution?
        data.Push(valobj->GetTypeImpl().GetCompilerType(false));
        break;
      }
      case sel_get_template_argument_type: {
        TYPE_CHECK(Type, UInt);
        auto index = data.Pop<uint64_t>();
        auto type = data.Pop<CompilerType>();
        // FIXME: There is more code in SBType::GetTemplateArgumentType().
        data.Push(type.GetTypeTemplateArgument(index, true));
        break;
      }
      case sel_get_value: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        data.Push(std::string(valobj->GetValueAsCString()));
        break;
      }
      case sel_get_value_as_unsigned: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        bool success;
        uint64_t val = valobj->GetValueAsUnsigned(0, &success);
        data.Push(val);
        if (!success)
          return sel_error("failed to get value");
        break;
      }
      case sel_get_value_as_signed: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        bool success;
        int64_t val = valobj->GetValueAsSigned(0, &success);
        data.Push(val);
        if (!success)
          return sel_error("failed to get value");
        break;
      }
      case sel_get_value_as_address: {
        TYPE_CHECK(Object);
        POP_VALOBJ(valobj);
        bool success;
        uint64_t addr = valobj->GetValueAsUnsigned(0, &success);
        if (!success)
          return sel_error("failed to get value");
        if (auto process_sp = valobj->GetProcessSP())
          addr = process_sp->FixDataAddress(addr);
        data.Push(addr);
        break;
      }
      case sel_cast: {
        TYPE_CHECK(Object, Type);
        auto type = data.Pop<CompilerType>();
        POP_VALOBJ(valobj);
        data.Push(valobj->Cast(type));
        break;
      }
      case sel_strlen: {
        TYPE_CHECK(String);
        data.Push((uint64_t)data.Pop<std::string>().size());
        break;
      }
      case sel_fmt: {
        TYPE_CHECK(String);
        if (auto error = FormatImpl(data))
          return error;
        break;
      }
      default:
        return sel_error("selector not implemented");
      }
      continue;
    }
    }
    return error("opcode not implemented");
  }
  return pc.takeError();
}
} // namespace FormatterBytecode

} // namespace lldb_private
