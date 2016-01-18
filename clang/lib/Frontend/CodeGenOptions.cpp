//===--- CodeGenOptions.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CodeGenOptions.h"
#include <string.h>

namespace clang {

CodeGenOptions::CodeGenOptions() {
#define CODEGENOPT(Name, Bits, Default) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) set##Name(Default);
#include "clang/Frontend/CodeGenOptions.def"

  RelocationModel = "pic";
  memcpy(CoverageVersion, "402*", 4);
}

}  // end namespace clang
