//===- SerializationFormat.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"

using namespace clang::ssaf;

char SerializationFormat::ID = 0;
SerializationFormat::SerializationFormat(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : FS(FS) {}
