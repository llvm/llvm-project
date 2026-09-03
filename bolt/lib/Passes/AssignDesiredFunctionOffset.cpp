//===- bolt/Passes/AssignDesiredFunctionOffset.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AssignDesiredFunctionOffset pass.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/AssignDesiredFunctionOffset.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryData.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;

cl::opt<std::string> FunctionLayoutFile(
    "function-layout-file",
    cl::desc("file populating the functions' desired output offset. Requires "
             "relocation mode."),
    cl::value_desc("filename"), cl::Hidden, cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

/// Mirror the lookup in ReorderFunctions.
static BinaryFunction *lookupFunction(BinaryContext &BC, StringRef Name) {
  BinaryData *BD = BC.getBinaryDataByName(Name);
  if (!BD) {
    for (uint32_t LocalID = 1;; ++LocalID) {
      BD = BC.getBinaryDataByName((Name + "/" + Twine(LocalID)).str());
      if (!BD)
        break;
      if (BinaryFunction *BF = BC.getFunctionForSymbol(BD->getSymbol()))
        return BF;
    }
    return nullptr;
  }
  return BC.getFunctionForSymbol(BD->getSymbol());
}

Error AssignDesiredFunctionOffset::runOnFunctions(BinaryContext &BC) {
  if (opts::FunctionLayoutFile.empty())
    return Error::success();

  if (!BC.HasRelocations) {
    BC.errs() << "BOLT-ERROR: --function-layout-file is not supported in "
                 "non-relocation mode\n";
    exit(1);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFile(opts::FunctionLayoutFile);
  if (std::error_code EC = MB.getError())
    return createStringError(EC, Twine("cannot open function layout file '") +
                                     opts::FunctionLayoutFile +
                                     "': " + EC.message());

  uint64_t Applied = 0;
  uint64_t NotFound = 0;
  uint64_t Malformed = 0;
  for (line_iterator LI(*MB.get(), /*SkipBlanks=*/true, /*CommentMarker=*/'#');
       !LI.is_at_eof(); ++LI) {
    StringRef Line = LI->trim();
    if (Line.empty())
      continue;

    StringRef Name, OffsetStr;
    std::tie(Name, OffsetStr) = Line.split(' ');
    Name = Name.trim();
    OffsetStr = OffsetStr.trim();

    uint64_t Offset;
    if (Name.empty() || OffsetStr.empty() ||
        OffsetStr.getAsInteger(/*Radix=*/0, Offset)) {
      BC.errs() << "BOLT-WARNING: --function-layout-file: malformed entry at "
                << opts::FunctionLayoutFile << ":" << LI.line_number() << "\n";
      ++Malformed;
      continue;
    }

    BinaryFunction *BF = lookupFunction(BC, Name);
    if (!BF) {
      if (opts::Verbosity >= 1)
        BC.errs() << "BOLT-WARNING: --function-layout-file: cannot find "
                     "function '"
                  << Name << "'\n";
      ++NotFound;
      continue;
    }

    if (!isAligned(BF->getMinAlign(), Offset)) {
      BC.errs() << "BOLT-WARNING: --function-layout-file: offset " << Offset
                << " for function '" << Name
                << "' is not aligned by the minimum function alignment ("
                << BF->getMinAlign().value() << "), skipping\n";
      ++Malformed;
      continue;
    }

    BF->setDesiredOffset(Offset);
    ++Applied;
  }

  BC.outs() << "BOLT-INFO: --function-layout-file: pinned " << Applied
            << " functions";
  if (NotFound)
    BC.outs() << ", " << NotFound << " not found";
  if (Malformed)
    BC.outs() << ", " << Malformed << " malformed/skipped";
  BC.outs() << "\n";

  return Error::success();
}

} // namespace bolt
} // namespace llvm
