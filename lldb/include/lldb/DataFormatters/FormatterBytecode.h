//===-- FormatterBytecode.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Symbol/CompilerType.h"

namespace lldb_private {

namespace FormatterBytecode {

enum DataType : uint8_t { Any, String, Int, UInt, Object, Type, Selector };

enum OpCodes : uint8_t {
#define DEFINE_OPCODE(OP, MNEMONIC, NAME) op_##NAME = OP,
#include "FormatterBytecode.def"
#undef DEFINE_OPCODE
};

enum Selectors : uint8_t {
#define DEFINE_SELECTOR(ID, NAME) sel_##NAME = ID,
#include "FormatterBytecode.def"
#undef DEFINE_SELECTOR
};

enum Signatures : uint8_t {
#define DEFINE_SIGNATURE(ID, NAME) sig_##NAME = ID,
#include "FormatterBytecode.def"
#undef DEFINE_SIGNATURE
};

using ControlStackElement = llvm::StringRef;
using DataStackElement =
    std::variant<std::string, uint64_t, int64_t, lldb::ValueObjectSP,
                 CompilerType, Selectors>;
struct DataStack : public std::vector<DataStackElement> {
  DataStack() = default;
  DataStack(lldb::ValueObjectSP initial_value)
      : std::vector<DataStackElement>({initial_value}) {}
  void Push(DataStackElement el) { push_back(el); }
  template <typename T> T Pop() {
    T el = std::get<T>(back());
    pop_back();
    return el;
  }
  DataStackElement PopAny() {
    DataStackElement el = back();
    pop_back();
    return el;
  }
};
llvm::Error Interpret(std::vector<ControlStackElement> &control,
                      DataStack &data, Selectors sel);
} // namespace FormatterBytecode

std::string toString(FormatterBytecode::OpCodes op);
std::string toString(FormatterBytecode::Selectors sel);
std::string toString(FormatterBytecode::Signatures sig);

} // namespace lldb_private
