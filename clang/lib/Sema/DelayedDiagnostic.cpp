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

DelayedDiagnostic
DelayedDiagnostic::makeAvailability(AvailabilityResult AD,
                                    SourceLocation Loc,
                                    const NamedDecl *D,
                                    const ObjCInterfaceDecl *UnknownObjCClass,
                                    const ObjCPropertyDecl  *ObjCProperty,
                                    StringRef Msg,
                                    bool ObjCPropertyAccess) {
  DelayedDiagnostic DD;
  switch (AD) {
  case AR_Deprecated:
    DD.Kind = Deprecation;
    break;
  case AR_Unavailable:
    DD.Kind = Unavailable;
    break;
  default:
    llvm_unreachable("partial diags should not be delayed");
  }
  DD.Triggered = false;
  DD.Loc = Loc;
  DD.DeprecationData.Decl = D;
  DD.DeprecationData.UnknownObjCClass = UnknownObjCClass;
  DD.DeprecationData.ObjCProperty = ObjCProperty;
  char *MessageData = nullptr;
  if (Msg.size()) {
    MessageData = new char [Msg.size()];
    memcpy(MessageData, Msg.data(), Msg.size());
  }

  DD.DeprecationData.Message = MessageData;
  DD.DeprecationData.MessageLen = Msg.size();
  DD.DeprecationData.ObjCPropertyAccess = ObjCPropertyAccess;
  return DD;
}

void DelayedDiagnostic::Destroy() {
  switch (static_cast<DDKind>(Kind)) {
  case Access: 
    getAccessData().~AccessedEntity(); 
    break;

  case Deprecation:
  case Unavailable:
    delete [] DeprecationData.Message;
    break;

  case ForbiddenType:
    break;
  }
}
