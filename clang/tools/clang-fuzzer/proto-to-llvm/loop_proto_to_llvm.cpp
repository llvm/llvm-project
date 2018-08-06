//==-- loop_proto_to_llvm.cpp - Protobuf-C++ conversion
//---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements functions for converting between protobufs and LLVM IR.
//
//
//===----------------------------------------------------------------------===//

#include "loop_proto_to_llvm.h"
#include "cxx_loop_proto.pb.h"

// The following is needed to convert protos in human-readable form
#include <google/protobuf/text_format.h>

#include <ostream>
#include <sstream>

namespace clang_fuzzer {

// Forward decls
std::string BinopToString(std::ostream &os, const BinaryOp &x);
std::string StateSeqToString(std::ostream &os, const StatementSeq &x);

// Counter variable to generate new LLVM IR variable names and wrapper function
std::string get_var() {
  static int ctr = 0;
  return "%var" + std::to_string(ctr++);
}

// Proto to LLVM.

std::string ConstToString(const Const &x) {
  return std::to_string(x.val());
}
std::string VarRefToString(std::ostream &os, const VarRef &x) {
  std::string arr;
  switch(x.arr()) {
  case VarRef::ARR_A:
    arr = "%a";
    break;
  case VarRef::ARR_B:
    arr = "%b";
    break;
  case VarRef::ARR_C:
    arr = "%c";
    break;
  }
  std::string ptr_var = get_var();
  os << ptr_var << " = getelementptr i32, i32* " << arr << ", i64 %ct\n";
  return ptr_var;
}
std::string RvalueToString(std::ostream &os, const Rvalue &x) {
  if(x.has_cons())
    return ConstToString(x.cons());
  if(x.has_binop())
    return BinopToString(os, x.binop());
  if(x.has_varref()) {
    std::string var_ref = VarRefToString(os, x.varref());
    std::string val_var = get_var();
    os << val_var << " = load i32, i32* " << var_ref << "\n";
    return val_var;
  }
  return "1";

}
std::string BinopToString(std::ostream &os, const BinaryOp &x) {
  std::string left = RvalueToString(os, x.left());
  std::string right = RvalueToString(os, x.right());
  std::string op;
  switch (x.op()) {
  case BinaryOp::PLUS:
    op = "add";
    break;
  case BinaryOp::MINUS:
    op = "sub";
    break;
  case BinaryOp::MUL:
    op = "mul";
    break;
  case BinaryOp::XOR:
    op = "xor";
    break;
  case BinaryOp::AND:
    op = "and";
    break;
  case BinaryOp::OR:
    op = "or";
    break;
  // Support for Boolean operators will be added later
  case BinaryOp::EQ:
  case BinaryOp::NE:
  case BinaryOp::LE:
  case BinaryOp::GE:
  case BinaryOp::LT:
  case BinaryOp::GT:
    op = "add";
    break;
  }
  std::string val_var = get_var();
  os << val_var << " = " << op << " i32 " << left << ", " << right << "\n";
  return val_var;
}
std::ostream &operator<<(std::ostream &os, const AssignmentStatement &x) {
  std::string rvalue = RvalueToString(os, x.rvalue());
  std::string var_ref = VarRefToString(os, x.varref());
  return os << "store i32 " << rvalue << ", i32* " << var_ref << "\n";
}
std::ostream &operator<<(std::ostream &os, const Statement &x) {
  return os << x.assignment();
}
std::ostream &operator<<(std::ostream &os, const StatementSeq &x) {
  for (auto &st : x.statements()) {
    os << st;
  }
  return os;
}
std::ostream &operator<<(std::ostream &os, const LoopFunction &x) {
  return os << "define void @foo(i32* %a, i32* %b, i32* noalias %c, i64 %s) {\n"
            << "%i = alloca i64\n"
            << "store i64 0, i64* %i\n"
            << "br label %loop\n\n"
            << "loop:\n"
            << "%ct = load i64, i64* %i\n"
            << "%comp = icmp eq i64 %ct, %s\n"
            << "br i1 %comp, label %endloop, label %body\n\n"
            << "body:\n"
            << x.statements()
            << "%z = add i64 1, %ct\n"
            << "store i64 %z, i64* %i\n"
            << "br label %loop\n\n"
            << "endloop:\n"
            << "ret void\n}\n";
}

// ---------------------------------

std::string LoopFunctionToLLVMString(const LoopFunction &input) {
  std::ostringstream os;
  os << input;
  return os.str();
}
std::string LoopProtoToLLVM(const uint8_t *data, size_t size) {
  LoopFunction message;
  if (!message.ParsePartialFromArray(data, size))
    return "#error invalid proto\n";
  return LoopFunctionToLLVMString(message);
}

} // namespace clang_fuzzer
