//==-- ObjCRetainCount.h - Retain count summaries for Cocoa -------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the core data structures for retain count "summaries"
//  for Objective-C and Core Foundation APIs.  These summaries are used
//  by the static analyzer to summarize the retain/release effects of
//  function and method calls.  This drives a path-sensitive typestate
//  analysis in the static analyzer, but can also potentially be used by
//  other clients.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CHECKERS_OBJCRETAINCOUNT_H
#define LLVM_CLANG_STATICANALYZER_CHECKERS_OBJCRETAINCOUNT_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class FunctionDecl;
class ObjCMethodDecl;

namespace ento { namespace objc_retain {

/// An ArgEffect summarizes the retain count behavior on an argument or receiver
/// to a function or method.
enum ArgEffect {
  /// There is no effect.
  DoNothing,

  /// The argument is treated as if an -autorelease message had been sent to
  /// the referenced object.
  Autorelease,

  /// The argument is treated as if an -dealloc message had been sent to
  /// the referenced object.
  Dealloc,

  /// The argument has its reference count decreased by 1.  This is as
  /// if CFRelease has been called on the argument.
  DecRef,

  /// The argument has its reference count decreased by 1.  This is as
  /// if a -release message has been sent to the argument.  This differs
  /// in behavior from DecRef when GC is enabled.
  DecRefMsg,

  /// The argument has its reference count decreased by 1 to model
  /// a transferred bridge cast under ARC.
  DecRefBridgedTransferred,

  /// The argument has its reference count increased by 1.  This is as
  /// if a -retain message has been sent to the argument.  This differs
  /// in behavior from IncRef when GC is enabled.
  IncRefMsg,

  /// The argument has its reference count increased by 1.  This is as
  /// if CFRetain has been called on the argument.
  IncRef,

  /// The argument acts as if has been passed to CFMakeCollectable, which
  /// transfers the object to the Garbage Collector under GC.
  MakeCollectable,

  /// The argument is a pointer to a retain-counted object; on exit, the new
  /// value of the pointer is a +0 value or NULL.
  UnretainedOutParameter,

  /// The argument is a pointer to a retain-counted object; on exit, the new
  /// value of the pointer is a +1 value or NULL.
  RetainedOutParameter,

  /// The argument is treated as potentially escaping, meaning that
  /// even when its reference count hits 0 it should be treated as still
  /// possibly being alive as someone else *may* be holding onto the object.
  MayEscape,

  /// All typestate tracking of the object ceases.  This is usually employed
  /// when the effect of the call is completely unknown.
  StopTracking,

  /// All typestate tracking of the object ceases.  Unlike StopTracking,
  /// this is also enforced when the method body is inlined.
  ///
  /// In some cases, we obtain a better summary for this checker
  /// by looking at the call site than by inlining the function.
  /// Signifies that we should stop tracking the symbol even if
  /// the function is inlined.
  StopTrackingHard,

  /// Performs the combined functionality of DecRef and StopTrackingHard.
  ///
  /// The models the effect that the called function decrements the reference
  /// count of the argument and all typestate tracking on that argument
  /// should cease.
  DecRefAndStopTrackingHard,

  /// Performs the combined functionality of DecRefMsg and StopTrackingHard.
  ///
  /// The models the effect that the called function decrements the reference
  /// count of the argument and all typestate tracking on that argument
  /// should cease.
  DecRefMsgAndStopTrackingHard
};

/// RetEffect summarizes a call's retain/release behavior with respect
/// to its return value.
class RetEffect {
public:
  enum Kind {
    /// Indicates that no retain count information is tracked for
    /// the return value.
    NoRet,
    /// Indicates that the returned value is an owned (+1) symbol.
    OwnedSymbol,
    /// Indicates that the returned value is an object with retain count
    /// semantics but that it is not owned (+0).  This is the default
    /// for getters, etc.
    NotOwnedSymbol,
    /// Indicates that the object is not owned and controlled by the
    /// Garbage collector.
    GCNotOwnedSymbol,
    /// Indicates that the return value is an owned object when the
    /// receiver is also a tracked object.
    OwnedWhenTrackedReceiver,
    // Treat this function as returning a non-tracked symbol even if
    // the function has been inlined. This is used where the call
    // site summary is more presise than the summary indirectly produced
    // by inlining the function
    NoRetHard
  };
  
  /// Determines the object kind of a tracked object.
  enum ObjKind {
    /// Indicates that the tracked object is a CF object.  This is
    /// important between GC and non-GC code.
    CF,
    /// Indicates that the tracked object is an Objective-C object.
    ObjC,
    /// Indicates that the tracked object could be a CF or Objective-C object.
    AnyObj,
    /// Indicates that the tracked object is a generalized object.
    Generalized
  };

private:
  Kind K;
  ObjKind O;
  
  RetEffect(Kind k, ObjKind o = AnyObj) : K(k), O(o) {}
  
public:
  Kind getKind() const { return K; }
  
  ObjKind getObjKind() const { return O; }
  
  bool isOwned() const {
    return K == OwnedSymbol || K == OwnedWhenTrackedReceiver;
  }
  
  bool notOwned() const {
    return K == NotOwnedSymbol;
  }
  
  bool operator==(const RetEffect &Other) const {
    return K == Other.K && O == Other.O;
  }
  
  static RetEffect MakeOwnedWhenTrackedReceiver() {
    return RetEffect(OwnedWhenTrackedReceiver, ObjC);
  }
  
  static RetEffect MakeOwned(ObjKind o) {
    return RetEffect(OwnedSymbol, o);
  }
  static RetEffect MakeNotOwned(ObjKind o) {
    return RetEffect(NotOwnedSymbol, o);
  }
  static RetEffect MakeGCNotOwned() {
    return RetEffect(GCNotOwnedSymbol, ObjC);
  }
  static RetEffect MakeNoRet() {
    return RetEffect(NoRet);
  }
  static RetEffect MakeNoRetHard() {
    return RetEffect(NoRetHard);
  }
};

/// Encapsulates the retain count semantics on the arguments, return value,
/// and receiver (if any) of a function/method call.
///
/// Note that construction of these objects is not highly efficient.  That
/// is okay for clients where creating these objects isn't really a bottleneck.
/// The purpose of the API is to provide something simple.  The actual
/// static analyzer checker that implements retain/release typestate
/// tracking uses something more efficient.
class CallEffects {
  llvm::SmallVector<ArgEffect, 10> Args;
  RetEffect Ret;
  ArgEffect Receiver;

  CallEffects(const RetEffect &R) : Ret(R) {}

public:
  /// Returns the argument effects for a call.
  ArrayRef<ArgEffect> getArgs() const { return Args; }

  /// Returns the effects on the receiver.
  ArgEffect getReceiver() const { return Receiver; }

  /// Returns the effect on the return value.
  RetEffect getReturnValue() const { return Ret; }

  /// Return the CallEfect for a given Objective-C method.
  static CallEffects getEffect(const ObjCMethodDecl *MD);

  /// Return the CallEfect for a given C/C++ function.
  static CallEffects getEffect(const FunctionDecl *FD);
};

}}}

#endif

