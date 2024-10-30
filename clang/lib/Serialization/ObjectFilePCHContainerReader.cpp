//===--- ObjectFilePCHContainerReader.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"

using namespace clang;

ArrayRef<StringRef> ObjectFilePCHContainerReader::getFormats() const {
  static StringRef Formats[] = {"obj", "raw"};
  return Formats;
}

StringRef
ObjectFilePCHContainerReader::ExtractPCH(llvm::MemoryBufferRef Buffer) const {
  StringRef PCH;
  auto OFOrErr = llvm::object::ObjectFile::createObjectFile(Buffer);
  if (OFOrErr) {
    auto &OF = OFOrErr.get();
    bool IsCOFF = isa<llvm::object::COFFObjectFile>(*OF);
    // Find the clang AST section in the container.
    for (auto &Section : OF->sections()) {
      StringRef Name;
      if (Expected<StringRef> NameOrErr = Section.getName())
        Name = *NameOrErr;
      else
        consumeError(NameOrErr.takeError());

      if ((!IsCOFF && Name == "__clangast") || (IsCOFF && Name == "clangast")) {
        if (Expected<StringRef> E = Section.getContents())
          return *E;
        else {
          handleAllErrors(E.takeError(), [&](const llvm::ErrorInfoBase &EIB) {
            EIB.log(llvm::errs());
          });
          return "";
        }
      }
    }
  }
  handleAllErrors(OFOrErr.takeError(), [&](const llvm::ErrorInfoBase &EIB) {
    if (EIB.convertToErrorCode() ==
        llvm::object::object_error::invalid_file_type)
      // As a fallback, treat the buffer as a raw AST.
      PCH = Buffer.getBuffer();
    else
      EIB.log(llvm::errs());
  });
  return PCH;
}
