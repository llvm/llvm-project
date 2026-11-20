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
#include "llvm/Support/MemoryBuffer.h"

using namespace clang;

namespace {
/// A buffer exposing a sub-range of an owned buffer.
class SubMemoryBuffer : public llvm::MemoryBuffer {
  std::unique_ptr<llvm::MemoryBuffer> Underlying;

public:
  SubMemoryBuffer(std::unique_ptr<llvm::MemoryBuffer> Underlying,
                  StringRef Range)
      : Underlying(std::move(Underlying)) {
    init(Range.begin(), Range.end(), /*RequiresNullTerminator=*/false);
  }

  StringRef getBufferIdentifier() const override {
    return Underlying->getBufferIdentifier();
  }

  BufferKind getBufferKind() const override {
    return Underlying->getBufferKind();
  }
};
} // namespace

ArrayRef<StringRef> ObjectFilePCHContainerReader::getFormats() const {
  static StringRef Formats[] = {"obj", "raw"};
  return Formats;
}

/// Returns the serialized AST inside the PCH container Buffer.
static StringRef extractPCH(llvm::MemoryBufferRef Buffer) {
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

std::unique_ptr<llvm::MemoryBuffer> ObjectFilePCHContainerReader::ExtractPCH(
    std::unique_ptr<llvm::MemoryBuffer> Buffer) const {
  StringRef Extracted = extractPCH(*Buffer);
  // As a fallback, the buffer is treated as a raw AST (see extractPCH()). Use
  // it as-is, there's no container to unwrap.
  if (Extracted.begin() == Buffer->getBufferStart() &&
      Extracted.end() == Buffer->getBufferEnd())
    return Buffer;
  return std::make_unique<SubMemoryBuffer>(std::move(Buffer), Extracted);
}
