//===--- Utility.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide some utility classes for use in the demangler.
// There are two copies of this file in the source tree.  The one in libcxxabi
// is the original and the one in llvm is the copy.  Use cp-to-llvm.sh to update
// the copy.  See README.txt for more details.
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "DemangleConfig.h"

DEMANGLE_NAMESPACE_BEGIN

bool FunctionNameInfo::startedPrintingArguments() const {
  return ArgumentLocs.first > 0;
}

bool FunctionNameInfo::shouldTrack(OutputBuffer &OB) const {
  if (!OB.isPrintingTopLevelFunctionType())
    return false;

  if (OB.isGtInsideTemplateArgs())
    return false;

  if (startedPrintingArguments())
    return false;

  return true;
}

bool FunctionNameInfo::canFinalize(OutputBuffer &OB) const {
  if (!OB.isPrintingTopLevelFunctionType())
    return false;

  if (OB.isGtInsideTemplateArgs())
    return false;

  if (!startedPrintingArguments())
    return false;

  return true;
}

void FunctionNameInfo::updateBasenameEnd(OutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  BasenameLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::updateScopeStart(OutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  ScopeLocs.first = OB.getCurrentPosition();
}

void FunctionNameInfo::updateScopeEnd(OutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  ScopeLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::finalizeArgumentEnd(OutputBuffer &OB) {
  if (!canFinalize(OB))
    return;

  OB.FunctionInfo.ArgumentLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::finalizeStart(OutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  OB.FunctionInfo.ArgumentLocs.first = OB.getCurrentPosition();

  // If nothing has set the end of the basename yet (for example when
  // printing templates), then the beginning of the arguments is the end of
  // the basename.
  if (BasenameLocs.second == 0)
    OB.FunctionInfo.BasenameLocs.second = OB.getCurrentPosition();

  DEMANGLE_ASSERT(!shouldTrack(OB), "");
  DEMANGLE_ASSERT(canFinalize(OB), "");
}

void FunctionNameInfo::finalizeEnd(OutputBuffer &OB) {
  if (!canFinalize(OB))
    return;

  if (ScopeLocs.first > OB.FunctionInfo.ScopeLocs.second)
    ScopeLocs.second = OB.FunctionInfo.ScopeLocs.first;
  BasenameLocs.first = OB.FunctionInfo.ScopeLocs.second;
}

bool FunctionNameInfo::hasBasename() const {
  return BasenameLocs.first != BasenameLocs.second && BasenameLocs.second > 0;
}

ScopedOverride<unsigned> OutputBuffer::enterFunctionTypePrinting() {
  return {FunctionPrintingDepth, FunctionPrintingDepth + 1};
}

DEMANGLE_NAMESPACE_END
