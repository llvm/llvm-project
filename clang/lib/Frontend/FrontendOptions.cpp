//===--- FrontendOptions.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    .Case("i", InputKind(InputKind::C, true))
    .Case("ii", InputKind(InputKind::CXX, true))
    .Case("cui", InputKind(InputKind::CUDA, true))
    .Case("m", InputKind::ObjC)
    .Case("mi", InputKind(InputKind::ObjC, true))
    .Cases("mm", "M", InputKind::ObjCXX)
    .Case("mii", InputKind(InputKind::ObjCXX, true))
    .Cases("C", "cc", "cp", InputKind::CXX)
    .Cases("cpp", "CPP", "c++", "cxx", "hpp", InputKind::CXX)
    .Case("cppm", InputKind::CXX)
    .Case("iim", InputKind(InputKind::CXX, true))
    .Case("cl", InputKind::OpenCL)
    .Case("cu", InputKind::CUDA)
    .Cases("ll", "bc", InputKind::LLVM_IR)
    .Default(InputKind::Unknown);
}
