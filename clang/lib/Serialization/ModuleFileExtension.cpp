//===-- ModuleFileExtension.cpp - Module File Extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Serialization/ModuleFileExtension.h"
#include "llvm/ADT/Hashing.h"
using namespace clang;

char ModuleFileExtension::ID = 0;

ModuleFileExtension::~ModuleFileExtension() { }

llvm::hash_code ModuleFileExtension::hashExtension(llvm::hash_code Code) const {
  return Code;
}

ModuleFileExtensionWriter::~ModuleFileExtensionWriter() { }

ModuleFileExtensionReader::~ModuleFileExtensionReader() { }
