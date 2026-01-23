//===- Obsolete.cpp - Obsolete libclang functions and types -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// This file contains libclang symbols whose underlying functionality has been
// removed from Clang, but which need to be kept around so as to retain ABI
// compatibility.
//
//===--------------------------------------------------------------------===//

#include "clang-c/CXString.h"
#include "clang-c/Index.h"
#include "clang-c/Platform.h"
#include "llvm/Support/raw_ostream.h"

extern "C" {

// The functions below used to be part of the C API for ARCMigrate, which has
// since been removed from Clang; they already used to print an error if Clang
// was compiled without arcmt support, so we continue doing so.
CXRemapping clang_getRemappings(const char *) {
  llvm::errs() << "error: ARCMigrate has been removed from Clang";
  return nullptr;
}

CXRemapping clang_getRemappingsFromFileList(const char **, unsigned) {
  llvm::errs() << "error: ARCMigrate has been removed from Clang";
  return nullptr;
}

unsigned clang_remap_getNumFiles(CXRemapping) {
  llvm::errs() << "error: ARCMigrate has been removed from Clang";
  return 0;
}

void clang_remap_getFilenames(CXRemapping, unsigned, CXString *, CXString *) {
  llvm::errs() << "error: ARCMigrate has been removed from Clang";
}

void clang_remap_dispose(CXRemapping) {
  llvm::errs() << "error: ARCMigrate has been removed from Clang";
}

} // extern "C"
