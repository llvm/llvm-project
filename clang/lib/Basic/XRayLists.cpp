//===--- XRayFunctionFilter.cpp - XRay automatic-attribution --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// User-provided filters for always/never XRay instrumenting certain functions.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/XRayLists.h"

using namespace clang;

XRayFunctionFilter::XRayFunctionFilter(
    ArrayRef<std::string> AlwaysInstrumentPaths,
    ArrayRef<std::string> NeverInstrumentPaths, SourceManager &SM)
    : AlwaysInstrument(
          llvm::SpecialCaseList::createOrDie(AlwaysInstrumentPaths)),
      NeverInstrument(llvm::SpecialCaseList::createOrDie(NeverInstrumentPaths)),
      SM(SM) {}

XRayFunctionFilter::ImbueAttribute
XRayFunctionFilter::shouldImbueFunction(StringRef FunctionName) const {
  // First apply the always instrument list, than if it isn't an "always" see
  // whether it's treated as a "never" instrument function.
  if (AlwaysInstrument->inSection("fun", FunctionName))
    return ImbueAttribute::ALWAYS;
  if (NeverInstrument->inSection("fun", FunctionName))
    return ImbueAttribute::NEVER;
  return ImbueAttribute::NONE;
}

XRayFunctionFilter::ImbueAttribute
XRayFunctionFilter::shouldImbueFunctionsInFile(StringRef Filename,
                                               StringRef Category) const {
  if (AlwaysInstrument->inSection("src", Filename, Category))
    return ImbueAttribute::ALWAYS;
  if (NeverInstrument->inSection("src", Filename, Category))
    return ImbueAttribute::NEVER;
  return ImbueAttribute::NONE;
}

XRayFunctionFilter::ImbueAttribute
XRayFunctionFilter::shouldImbueLocation(SourceLocation Loc,
                                        StringRef Category) const {
  if (!Loc.isValid())
    return ImbueAttribute::NONE;
  return this->shouldImbueFunctionsInFile(SM.getFilename(SM.getFileLoc(Loc)),
                                          Category);
}
