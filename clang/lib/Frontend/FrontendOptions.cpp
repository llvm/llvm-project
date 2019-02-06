//===- FrontendOptions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendOptions.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

InputKind FrontendOptions::getInputKindForExtension(StringRef Extension) {
  return llvm::StringSwitch<InputKind>(Extension)
    .Cases("ast", "pcm", InputKind(InputKind::Unknown, InputKind::Precompiled))
    .Case("c", InputKind::C)
    .Cases("S", "s", InputKind::Asm)
    .Case("i", InputKind(InputKind::C).getPreprocessed())
    .Case("ii", InputKind(InputKind::CXX).getPreprocessed())
    .Case("cui", InputKind(InputKind::CUDA).getPreprocessed())
    .Case("m", InputKind::ObjC)
    .Case("mi", InputKind(InputKind::ObjC).getPreprocessed())
    .Cases("mm", "M", InputKind::ObjCXX)
    .Case("mii", InputKind(InputKind::ObjCXX).getPreprocessed())
    .Cases("C", "cc", "cp", InputKind::CXX)
    .Cases("cpp", "CPP", "c++", "cxx", "hpp", InputKind::CXX)
    .Case("cppm", InputKind::CXX)
    .Case("iim", InputKind(InputKind::CXX).getPreprocessed())
    .Case("cl", InputKind::OpenCL)
    .Case("cu", InputKind::CUDA)
    .Cases("ll", "bc", InputKind::LLVM_IR)
    .Default(InputKind::Unknown);
}
