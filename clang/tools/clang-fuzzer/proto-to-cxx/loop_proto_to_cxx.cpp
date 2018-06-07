//==-- loop_proto_to_cxx.cpp - Protobuf-C++ conversion ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements functions for converting between protobufs and C++. Extends
// proto_to_cxx.cpp by wrapping all the generated C++ code in a single for
// loop. Also coutputs a different function signature that includes a
// size_t parameter for the loop to use.
//
//===----------------------------------------------------------------------===//

#include "cxx_loop_proto.pb.h"
#include "proto_to_cxx.h"

// The following is needed to convert protos in human-readable form
#include <google/protobuf/text_format.h>

#include <ostream>
#include <sstream>

namespace clang_fuzzer {

// Forward decls.
std::ostream &operator<<(std::ostream &os, const BinaryOp &x);
std::ostream &operator<<(std::ostream &os, const StatementSeq &x);

// Proto to C++.
std::ostream &operator<<(std::ostream &os, const Const &x) {
  return os << "(" << x.val() << ")";
}
std::ostream &operator<<(std::ostream &os, const VarRef &x) {
  if (x.is_loop_var()) {
    return os << "a[loop_ctr]";
  } else {
    return os << "a[" << static_cast<uint32_t>(x.varnum()) << " % s]";
  }
}
std::ostream &operator<<(std::ostream &os, const Lvalue &x) {
  return os << x.varref();
}
std::ostream &operator<<(std::ostream &os, const Rvalue &x) {
  if (x.has_varref())
    return os << x.varref();
  if (x.has_cons())
    return os << x.cons();
  if (x.has_binop())
    return os << x.binop();
  return os << "1";
}
std::ostream &operator<<(std::ostream &os, const BinaryOp &x) {
  os << "(" << x.left();
  switch (x.op()) {
  case BinaryOp::PLUS:
    os << "+";
    break;
  case BinaryOp::MINUS:
    os << "-";
    break;
  case BinaryOp::MUL:
    os << "*";
    break;
  case BinaryOp::DIV:
    os << "/";
    break;
  case BinaryOp::MOD:
    os << "%";
    break;
  case BinaryOp::XOR:
    os << "^";
    break;
  case BinaryOp::AND:
    os << "&";
    break;
  case BinaryOp::OR:
    os << "|";
    break;
  case BinaryOp::EQ:
    os << "==";
    break;
  case BinaryOp::NE:
    os << "!=";
    break;
  case BinaryOp::LE:
    os << "<=";
    break;
  case BinaryOp::GE:
    os << ">=";
    break;
  case BinaryOp::LT:
    os << "<";
    break;
  case BinaryOp::GT:
    os << ">";
    break;
  }
  return os << x.right() << ")";
}
std::ostream &operator<<(std::ostream &os, const AssignmentStatement &x) {
  return os << x.lvalue() << "=" << x.rvalue();
}
std::ostream &operator<<(std::ostream &os, const IfElse &x) {
  return os << "if (" << x.cond() << "){\n"
            << x.if_body() << "} else { \n"
            << x.else_body() << "}\n";
}
std::ostream &operator<<(std::ostream &os, const While &x) {
  return os << "while (" << x.cond() << "){\n" << x.body() << "}\n";
}
std::ostream &operator<<(std::ostream &os, const Statement &x) {
  if (x.has_assignment())
    return os << x.assignment() << ";\n";
  if (x.has_ifelse())
    return os << x.ifelse();
  if (x.has_while_loop())
    return os << x.while_loop();
  return os << "(void)0;\n";
}
std::ostream &operator<<(std::ostream &os, const StatementSeq &x) {
  for (auto &st : x.statements())
    os << st;
  return os;
}
std::ostream &operator<<(std::ostream &os, const LoopFunction &x) {
  return os << "void foo(int *a, size_t s) {\n"
            << "for (int loop_ctr = 0; loop_ctr < s; loop_ctr++){\n"
            << x.statements() << "}\n}\n";
}

// ---------------------------------

std::string LoopFunctionToString(const LoopFunction &input) {
  std::ostringstream os;
  os << input;
  return os.str();
}
std::string LoopProtoToCxx(const uint8_t *data, size_t size) {
  LoopFunction message;
  if (!message.ParsePartialFromArray(data, size))
    return "#error invalid proto, may not be binary encoded\n";
  return LoopFunctionToString(message);
}

} // namespace clang_fuzzer
