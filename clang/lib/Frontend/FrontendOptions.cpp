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

InputKind FrontendOptions::getInputKindForExtension(llvm::StringRef Extension) {
  return llvm::StringSwitch<InputKind>(Extension)
    .Case("ast", IK_AST)
    .Case("c", IK_C)
    .Cases("S", "s", IK_Asm)
    .Case("i", IK_PreprocessedC)
    .Case("ii", IK_PreprocessedCXX)
    .Case("m", IK_ObjC)
    .Case("mi", IK_PreprocessedObjC)
    .Cases("mm", "M", IK_ObjCXX)
    .Case("mii", IK_PreprocessedObjCXX)
    .Case("C", IK_CXX)
    .Cases("C", "cc", "cp", IK_CXX)
    .Cases("cpp", "CPP", "c++", "cxx", "hpp", IK_CXX)
    .Case("cl", IK_OpenCL)
    .Cases("ll", "bc", IK_LLVM_IR)
    .Default(IK_C);
}
