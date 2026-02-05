//===- MockSerializationFormat.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Registries/MockSerializationFormat.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

using namespace clang;
using namespace ssaf;

MockSerializationFormat::MockSerializationFormat(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : SerializationFormat(FS) {}

TUSummary MockSerializationFormat::readTUSummary(llvm::StringRef Path) {
  // TODO: Implement this.
  BuildNamespace NS(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummary Summary(NS);
  return Summary;
}

void MockSerializationFormat::writeTUSummary(const TUSummary &Summary,
                                             llvm::StringRef OutputDir) {
  // TODO: Implement this.
}

static SerializationFormatRegistry::Add<MockSerializationFormat>
    RegisterFormat("MockSerializationFormat",
                   "A serialization format for testing");
