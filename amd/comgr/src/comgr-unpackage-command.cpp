//===- comgr-unpackage-command.cpp - UnpackageCommand implementation
//--------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the UnpackageCommand, which extracts offload binaries
/// from a package via llvm::object::OffloadFile::extractOffloadBinaries().
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include <comgr-unpackage-command.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace COMGR {
using namespace llvm;

amd_comgr_status_t UnpackageCommand::execute(raw_ostream &LogS) {
  DenseMap<object::OffloadFile::TargetID, StringRef> Worklist;
  for (const auto &[Output, Target] : zip_equal(OutputFileNames, TargetIDs)) {
    Worklist[Target] = Output;
  }

  for (const object::OffloadFile &File : Files) {
    const object::OffloadBinary *Binary = File.getBinary();
    object::OffloadFile::TargetID Target = File;

    // Prefer an exact match, then fall back to a compatible target. Note that
    // areTargetsCompatible() reports false for an exact match (it only flags
    // distinct-but-compatible targets), so the explicit lookup below is
    // required to handle equal targets. We track the matching Worklist entry by
    // iterator so it can be erased once written.
    auto Match = Worklist.find(Target);
    if (Match == Worklist.end())
      Match = find_if(Worklist, [Target](const auto &KV) {
        return object::areTargetsCompatible(Target, KV.first);
      });

    if (Match == Worklist.end())
      continue;

    StringRef Image = Binary->getImage();

    // create an output file descriptor
    StringRef OutputName = Match->second;
    std::error_code EC;
    raw_fd_ostream OutputFile(OutputName, EC, sys::fs::OF_None);
    if (EC) {
      return AMD_COMGR_STATUS_ERROR;
    }

    // write the packaged image into the output
    OutputFile << Image;
    OutputFile.flush();

    // erase the entry from Worklist (so that we can track if all expected
    // files have been unpackaged)
    Worklist.erase(Match);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace COMGR
