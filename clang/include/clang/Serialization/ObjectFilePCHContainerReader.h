//===-- Serialization/ObjectFilePCHContainerReader.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_OBJECTFILEPCHCONTAINERREADER_H
#define LLVM_CLANG_SERIALIZATION_OBJECTFILEPCHCONTAINERREADER_H

#include "clang/Serialization/PCHContainerOperations.h"

namespace clang {
/// A PCHContainerReader implementation that uses LLVM to
/// wraps Clang modules inside a COFF, ELF, or Mach-O container.
class ObjectFilePCHContainerReader : public PCHContainerReader {
  ArrayRef<StringRef> getFormats() const override;

  /// Returns the serialized AST inside the PCH container Buffer.
  StringRef ExtractPCH(llvm::MemoryBufferRef Buffer) const override;
};
} // namespace clang

#endif
