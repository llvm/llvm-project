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
#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "bolt-rtlib"

using namespace llvm;
using namespace bolt;

void RuntimeLibrary::anchor() {}

std::string RuntimeLibrary::getLibPath(StringRef ToolPath,
                                       StringRef LibFileName) {
  StringRef Dir = llvm::sys::path::parent_path(ToolPath);
  SmallString<128> LibPath = llvm::sys::path::parent_path(Dir);
  llvm::sys::path::append(LibPath, "lib");
  if (!llvm::sys::fs::exists(LibPath)) {
    // In some cases we install bolt binary into one level deeper in bin/,
    // we need to go back one more level to find lib directory.
    LibPath = llvm::sys::path::parent_path(llvm::sys::path::parent_path(Dir));
    llvm::sys::path::append(LibPath, "lib");
  }
  llvm::sys::path::append(LibPath, LibFileName);
  if (!llvm::sys::fs::exists(LibPath)) {
    errs() << "BOLT-ERROR: library not found: " << LibPath << "\n";
    exit(1);
  }
  return std::string(LibPath.str());
}

void RuntimeLibrary::loadLibrary(StringRef LibPath, RuntimeDyld &RTDyld) {
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
        RTDyld.loadObject(*Obj);
    }
    check_error(std::move(Err), B->getBufferIdentifier());
  } else if (Magic == file_magic::elf_relocatable ||
             Magic == file_magic::elf_shared_object) {
    std::unique_ptr<object::ObjectFile> Obj = cantFail(
        object::ObjectFile::createObjectFile(B.get()->getMemBufferRef()),
        "error creating in-memory object");
    RTDyld.loadObject(*Obj);
  } else {
    errs() << "BOLT-ERROR: unrecognized library format: " << LibPath << "\n";
    exit(1);
  }
}
