//===--- DelayedDiagnostic.cpp - Delayed declarator diagnostics -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DelayedDiagnostic class implementation, which
// is used to record diagnostics that are being conditionally produced
// during declarator parsing.
//
// This file also defines AccessedEntity.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/DelayedDiagnostic.h"
#include <string.h>
using namespace clang;
using namespace sema;

DelayedDiagnostic DelayedDiagnostic::makeDeprecation(SourceLocation Loc,
                                    const NamedDecl *D,
                                    const ObjCInterfaceDecl *UnknownObjCClass,
                                    const ObjCPropertyDecl  *ObjCProperty,
                                    StringRef Msg) {
  DelayedDiagnostic DD;
  DD.Kind = Deprecation;
  DD.Triggered = false;
  DD.Loc = Loc;
  DD.DeprecationData.Decl = D;
  DD.DeprecationData.UnknownObjCClass = UnknownObjCClass;
  DD.DeprecationData.ObjCProperty = ObjCProperty;
  char *MessageData = 0;
  if (Msg.size()) {
    MessageData = new char [Msg.size()];
    memcpy(MessageData, Msg.data(), Msg.size());
  }

  DD.DeprecationData.Message = MessageData;
  DD.DeprecationData.MessageLen = Msg.size();
  return DD;
}

void DelayedDiagnostic::Destroy() {
  switch (Kind) {
  case Access: 
    getAccessData().~AccessedEntity(); 
    break;

  case Deprecation: 
    delete [] DeprecationData.Message;
    break;

  case ForbiddenType:
    break;
  }
}
