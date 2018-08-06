//===-- ExampleClangProtoFuzzer.cpp - Fuzz Clang --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a function that runs Clang on a single
///  input and uses libprotobuf-mutator to find new inputs. This function is
///  then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "cxx_proto.pb.h"
#include "handle-cxx/handle_cxx.h"
#include "proto-to-cxx/proto_to_cxx.h"
#include "fuzzer-initialize/fuzzer_initialize.h"
#include "src/libfuzzer/libfuzzer_macro.h"

using namespace clang_fuzzer;

DEFINE_BINARY_PROTO_FUZZER(const Function& input) {
  auto S = FunctionToString(input);
  HandleCXX(S, GetCLArgs());
}
