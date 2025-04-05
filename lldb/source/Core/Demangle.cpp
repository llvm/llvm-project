//===-- Demangle.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Demangle.h"

using namespace llvm::itanium_demangle;

namespace lldb_private {

bool FunctionNameInfo::startedPrintingArguments() const {
  return ArgumentLocs.first > 0;
}

bool FunctionNameInfo::shouldTrack(TrackingOutputBuffer &OB) const {
  if (!OB.isPrintingTopLevelFunctionType())
    return false;

  if (OB.isGtInsideTemplateArgs())
    return false;

  if (startedPrintingArguments())
    return false;

  return true;
}

bool FunctionNameInfo::canFinalize(TrackingOutputBuffer &OB) const {
  if (!OB.isPrintingTopLevelFunctionType())
    return false;

  if (OB.isGtInsideTemplateArgs())
    return false;

  if (!startedPrintingArguments())
    return false;

  return true;
}

void FunctionNameInfo::updateBasenameEnd(TrackingOutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  BasenameLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::updateScopeStart(TrackingOutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  ScopeLocs.first = OB.getCurrentPosition();
}

void FunctionNameInfo::updateScopeEnd(TrackingOutputBuffer &OB) {
  if (!shouldTrack(OB))
    return;

  ScopeLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::finalizeArgumentEnd(TrackingOutputBuffer &OB) {
  if (!canFinalize(OB))
    return;

  OB.FunctionInfo.ArgumentLocs.second = OB.getCurrentPosition();
}

void FunctionNameInfo::finalizeStart(TrackingOutputBuffer &OB) {
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

void FunctionNameInfo::finalizeEnd(TrackingOutputBuffer &OB) {
  if (!canFinalize(OB))
    return;

  if (ScopeLocs.first > OB.FunctionInfo.ScopeLocs.second)
    ScopeLocs.second = OB.FunctionInfo.ScopeLocs.first;
  BasenameLocs.first = OB.FunctionInfo.ScopeLocs.second;
}

bool FunctionNameInfo::hasBasename() const {
  return BasenameLocs.first != BasenameLocs.second && BasenameLocs.second > 0;
}

ScopedOverride<unsigned> TrackingOutputBuffer::enterFunctionTypePrinting() {
  return {FunctionPrintingDepth, FunctionPrintingDepth + 1};
}

bool TrackingOutputBuffer::isPrintingTopLevelFunctionType() const {
  return FunctionPrintingDepth == 1;
}

void TrackingOutputBuffer::printLeft(const Node &N) {
  switch (N.getKind()) {
  case Node::KFunctionType:
    printLeftImpl(static_cast<const FunctionType &>(N));
    break;
  case Node::KFunctionEncoding:
    printLeftImpl(static_cast<const FunctionEncoding &>(N));
    break;
  case Node::KNestedName:
    printLeftImpl(static_cast<const NestedName &>(N));
    break;
  case Node::KNameWithTemplateArgs:
    printLeftImpl(static_cast<const NameWithTemplateArgs &>(N));
    break;
  default:
    N.printLeft(*this);
  }
}

void TrackingOutputBuffer::printRight(const Node &N) {
  switch (N.getKind()) {
  case Node::KFunctionType:
    printRightImpl(static_cast<const FunctionType &>(N));
    break;
  case Node::KFunctionEncoding:
    printRightImpl(static_cast<const FunctionEncoding &>(N));
    break;
  default:
    N.printRight(*this);
  }
}

void TrackingOutputBuffer::printLeftImpl(const FunctionType &N) {
  auto Scoped = enterFunctionTypePrinting();
  N.printLeft(*this);
}

void TrackingOutputBuffer::printRightImpl(const FunctionType &N) {
  auto Scoped = enterFunctionTypePrinting();
  N.printRight(*this);
}

void TrackingOutputBuffer::printLeftImpl(const FunctionEncoding &N) {
  auto Scoped = enterFunctionTypePrinting();

  const Node *Ret = N.getReturnType();
  if (Ret) {
    printLeft(*Ret);
    if (!Ret->hasRHSComponent(*this))
      *this += " ";
  }

  FunctionInfo.updateScopeStart(*this);

  N.getName()->print(*this);
}

void TrackingOutputBuffer::printRightImpl(const FunctionEncoding &N) {
  auto Scoped = enterFunctionTypePrinting();
  FunctionInfo.finalizeStart(*this);

  printOpen();
  N.getParams().printWithComma(*this);
  printClose();

  FunctionInfo.finalizeArgumentEnd(*this);

  const Node *Ret = N.getReturnType();

  if (Ret)
    printRight(*Ret);

  auto CVQuals = N.getCVQuals();
  auto RefQual = N.getRefQual();
  auto *Attrs = N.getAttrs();
  auto *Requires = N.getRequires();

  if (CVQuals & QualConst)
    *this += " const";
  if (CVQuals & QualVolatile)
    *this += " volatile";
  if (CVQuals & QualRestrict)
    *this += " restrict";
  if (RefQual == FrefQualLValue)
    *this += " &";
  else if (RefQual == FrefQualRValue)
    *this += " &&";
  if (Attrs != nullptr)
    Attrs->print(*this);
  if (Requires != nullptr) {
    *this += " requires ";
    Requires->print(*this);
  }

  FunctionInfo.finalizeEnd(*this);
}

void TrackingOutputBuffer::printLeftImpl(const NestedName &N) {
  N.Qual->print(*this);
  *this += "::";
  FunctionInfo.updateScopeEnd(*this);
  N.Name->print(*this);
  FunctionInfo.updateBasenameEnd(*this);
}

void TrackingOutputBuffer::printLeftImpl(const NameWithTemplateArgs &N) {
  N.Name->print(*this);
  FunctionInfo.updateBasenameEnd(*this);
  N.TemplateArgs->print(*this);
}

} // namespace lldb_private
