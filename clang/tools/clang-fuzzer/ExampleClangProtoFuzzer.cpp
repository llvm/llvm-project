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
/// \brief This file implements a function that runs Clang on a single
///  input and uses libprotobuf-mutator to find new inputs. This function is
///  then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "cxx_proto.pb.h"
#include "handle-cxx/handle_cxx.h"
#include "proto-to-cxx/proto_to_cxx.h"

#include "src/libfuzzer/libfuzzer_macro.h"

#include <cstring>

using namespace clang_fuzzer;

static std::vector<const char *> CLArgs;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  CLArgs.push_back("-O2");
  for (int I = 1; I < *argc; I++) {
    if (strcmp((*argv)[I], "-ignore_remaining_args=1") == 0) {
      for (I++; I < *argc; I++)
        CLArgs.push_back((*argv)[I]);
      break;
    }
  }
  return 0;
}

DEFINE_BINARY_PROTO_FUZZER(const Function& input) {
  auto S = FunctionToString(input);
  HandleCXX(S, CLArgs);
}
