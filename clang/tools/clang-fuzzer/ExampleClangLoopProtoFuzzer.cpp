//===-- ExampleClangLoopProtoFuzzer.cpp - Fuzz Clang ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///  This file implements a function that runs Clang on a single
///  input and uses libprotobuf-mutator to find new inputs. This function is
///  then linked into the Fuzzer library. This file differs from
///  ExampleClangProtoFuzzer in that it uses a different protobuf that includes
///  C++ code with a single for loop.
///
//===----------------------------------------------------------------------===//

#include "cxx_loop_proto.pb.h"
#include "fuzzer-initialize/fuzzer_initialize.h"
#include "handle-cxx/handle_cxx.h"
#include "proto-to-cxx/proto_to_cxx.h"
#include "src/libfuzzer/libfuzzer_macro.h"

using namespace clang_fuzzer;

DEFINE_BINARY_PROTO_FUZZER(const LoopFunction &input) {
  auto S = LoopFunctionToString(input);
  HandleCXX(S, GetCLArgs());
}
