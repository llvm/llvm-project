//===-- ExampleClangLLVMProtoFuzzer.cpp - Fuzz Clang ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///  This file implements a function that compiles a single LLVM IR string as
///  input and uses libprotobuf-mutator to find new inputs. This function is
///  then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "cxx_loop_proto.pb.h"
#include "fuzzer-initialize/fuzzer_initialize.h"
#include "handle-llvm/handle_llvm.h"
#include "proto-to-llvm/loop_proto_to_llvm.h"
#include "src/libfuzzer/libfuzzer_macro.h"

using namespace clang_fuzzer;

DEFINE_BINARY_PROTO_FUZZER(const LoopFunction &input) {
  auto S = LoopFunctionToLLVMString(input);
  HandleLLVM(S, GetCLArgs());
}
