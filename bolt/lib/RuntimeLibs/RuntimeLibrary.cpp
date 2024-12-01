//===- bolt/RuntimeLibs/RuntimeLibrary.cpp - Runtime Library --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RuntimeLibrary class.
//
//===----------------------------------------------------------------------===//

#include "bolt/RuntimeLibs/RuntimeLibrary.h"
#include "bolt/Core/Linker.h"
#include "bolt/RuntimeLibs/RuntimeLibraryVariables.inc"
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "bolt-rtlib"

using namespace llvm;
using namespace bolt;

void RuntimeLibrary::anchor() {}

std::string RuntimeLibrary::getLibPathByToolPath(StringRef ToolPath,
                                                 StringRef LibFileName) {
  StringRef Dir = llvm::sys::path::parent_path(ToolPath);
  SmallString<128> LibPath = llvm::sys::path::parent_path(Dir);
  llvm::sys::path::append(LibPath, "lib" LLVM_LIBDIR_SUFFIX);
  if (!llvm::sys::fs::exists(LibPath)) {
    // In some cases we install bolt binary into one level deeper in bin/,
    // we need to go back one more level to find lib directory.
    LibPath = llvm::sys::path::parent_path(llvm::sys::path::parent_path(Dir));
    llvm::sys::path::append(LibPath, "lib" LLVM_LIBDIR_SUFFIX);
  }
  llvm::sys::path::append(LibPath, LibFileName);
  return std::string(LibPath);
}

std::string RuntimeLibrary::getLibPathByInstalled(StringRef LibFileName) {
  SmallString<128> LibPath(CMAKE_INSTALL_FULL_LIBDIR);
  llvm::sys::path::append(LibPath, LibFileName);
  return std::string(LibPath);
}

std::string RuntimeLibrary::getLibPath(StringRef ToolPath,
                                       StringRef LibFileName) {
  if (llvm::sys::fs::exists(LibFileName)) {
    return std::string(LibFileName);
  }

  std::string ByTool = getLibPathByToolPath(ToolPath, LibFileName);
  if (llvm::sys::fs::exists(ByTool)) {
    return ByTool;
  }

  std::string ByInstalled = getLibPathByInstalled(LibFileName);
  if (llvm::sys::fs::exists(ByInstalled)) {
    return ByInstalled;
  }

  errs() << "BOLT-ERROR: library not found: " << ByTool << ", " << ByInstalled
         << ", or " << LibFileName << "\n";
  exit(1);
}

void RuntimeLibrary::loadLibrary(StringRef LibPath, BOLTLinker &Linker,
                                 BOLTLinker::SectionsMapper MapSections) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeBuf =
      MemoryBuffer::getFile(LibPath, false, false);
  check_error(MaybeBuf.getError(), LibPath);
  std::unique_ptr<MemoryBuffer> B = std::move(MaybeBuf.get());
  file_magic Magic = identify_magic(B->getBuffer());

  if (Magic == file_magic::archive) {
    Error Err = Error::success();
    object::Archive Archive(B.get()->getMemBufferRef(), Err);
    for (const object::Archive::Child &C : Archive.children(Err)) {
      std::unique_ptr<object::Binary> Bin = cantFail(C.getAsBinary());
      if (object::ObjectFile *Obj = dyn_cast<object::ObjectFile>(&*Bin))
        Linker.loadObject(Obj->getMemoryBufferRef(), MapSections);
    }
    check_error(std::move(Err), B->getBufferIdentifier());
  } else if (Magic == file_magic::elf_relocatable ||
             Magic == file_magic::elf_shared_object) {
    std::unique_ptr<object::ObjectFile> Obj = cantFail(
        object::ObjectFile::createObjectFile(B.get()->getMemBufferRef()),
        "error creating in-memory object");
    Linker.loadObject(Obj->getMemoryBufferRef(), MapSections);
  } else {
    errs() << "BOLT-ERROR: unrecognized library format: " << LibPath << "\n";
    exit(1);
  }
}
