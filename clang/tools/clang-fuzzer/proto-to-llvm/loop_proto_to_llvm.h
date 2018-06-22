//==-- loop_proto_to_llvm.h - Protobuf-C++ conversion ----------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines functions for converting between protobufs and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstddef>
#include <string>

namespace clang_fuzzer {
class LoopFunction;

std::string LoopFunctionToLLVMString(const LoopFunction &input);
std::string LoopProtoToLLVM(const uint8_t *data, size_t size);
}
