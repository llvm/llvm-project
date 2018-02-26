//==-- proto_to_cxx.h - Protobuf-C++ conversion ----------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines functions for converting between protobufs and C++.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstddef>
#include <string>

namespace clang_fuzzer {
class Function;
std::string FunctionToString(const Function &input);
std::string ProtoToCxx(const uint8_t *data, size_t size);
}
