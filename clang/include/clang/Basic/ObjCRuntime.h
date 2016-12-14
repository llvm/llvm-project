//===--- ObjCRuntime.h - Objective-C Runtime Configuration ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines types useful for describing an Objective-C runtime.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OBJCRUNTIME_H
#define LLVM_CLANG_BASIC_OBJCRUNTIME_H

#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

/// \brief The basic abstraction for the target Objective-C runtime.
class ObjCRuntime {
public:
  /// \brief The basic Objective-C runtimes that we know about.
  enum Kind {
    /// 'macosx' is the Apple-provided NeXT-derived runtime on Mac OS
    /// X platforms that use the non-fragile ABI; the version is a
    /// release of that OS.
    MacOSX,

    /// 'macosx-fragile' is the Apple-provided NeXT-derived runtime on
    /// Mac OS X platforms that use the fragile ABI; the version is a
    /// release of that OS.
    FragileMacOSX,

    /// 'ios' is the Apple-provided NeXT-derived runtime on iOS or the iOS
    /// simulator;  it is always non-fragile.  The version is a release
    /// version of iOS.
    iOS,

    /// 'watchos' is a variant of iOS for Apple's watchOS. The version
    /// is a release version of watchOS.
    WatchOS,

    /// 'gcc' is the Objective-C runtime shipped with GCC, implementing a
    /// fragile Objective-C ABI
    GCC,

    /// 'gnustep' is the modern non-fragile GNUstep runtime.
    GNUstep,

    /// 'objfw' is the Objective-C runtime included in ObjFW
    ObjFW
  };

private:
  Kind TheKind;
  VersionTuple Version;

public:
  /// A bogus initialization of the runtime.
  ObjCRuntime() : TheKind(MacOSX) {}

  ObjCRuntime(Kind kind, const VersionTuple &version)
    : TheKind(kind), Version(version) {}

  void set(Kind kind, VersionTuple version) {
    TheKind = kind;
    Version = version;
  }

  Kind getKind() const { return TheKind; }
  const VersionTuple &getVersion() const { return Version; }

  /// \brief Does this runtime follow the set of implied behaviors for a
  /// "non-fragile" ABI?
  bool isNonFragile() const {
    switch (getKind()) {
    case FragileMacOSX: return false;
    case GCC: return false;
    case MacOSX: return true;
    case GNUstep: return true;
    case ObjFW: return true;
    case iOS: return true;
    case WatchOS: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// The inverse of isNonFragile():  does this runtime follow the set of
  /// implied behaviors for a "fragile" ABI?
  bool isFragile() const { return !isNonFragile(); }

  /// The default dispatch mechanism to use for the specified architecture
  bool isLegacyDispatchDefaultForArch(llvm::Triple::ArchType Arch) {
    // The GNUstep runtime uses a newer dispatch method by default from
    // version 1.6 onwards
    if (getKind() == GNUstep && getVersion() >= VersionTuple(1, 6)) {
      if (Arch == llvm::Triple::arm ||
          Arch == llvm::Triple::x86 ||
          Arch == llvm::Triple::x86_64)
        return false;
    }
    else if ((getKind() ==  MacOSX) && isNonFragile() &&
             (getVersion() >= VersionTuple(10, 0)) &&
             (getVersion() < VersionTuple(10, 6)))
        return Arch != llvm::Triple::x86_64;
    // Except for deployment target of 10.5 or less,
    // Mac runtimes use legacy dispatch everywhere now.
    return true;
  }

  /// \brief Is this runtime basically of the GNU family of runtimes?
  bool isGNUFamily() const {
    switch (getKind()) {
    case FragileMacOSX:
    case MacOSX:
    case iOS:
    case WatchOS:
      return false;
    case GCC:
    case GNUstep:
    case ObjFW:
      return true;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Is this runtime basically of the NeXT family of runtimes?
  bool isNeXTFamily() const {
    // For now, this is just the inverse of isGNUFamily(), but that's
    // not inherently true.
    return !isGNUFamily();
  }

  /// \brief Does this runtime allow ARC at all?
  bool allowsARC() const {
    switch (getKind()) {
    case FragileMacOSX:
      // No stub library for the fragile runtime.
      return getVersion() >= VersionTuple(10, 7);
    case MacOSX: return true;
    case iOS: return true;
    case WatchOS: return true;
    case GCC: return false;
    case GNUstep: return true;
    case ObjFW: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Does this runtime natively provide the ARC entrypoints? 
  ///
  /// ARC cannot be directly supported on a platform that does not provide
  /// these entrypoints, although it may be supportable via a stub
  /// library.
  bool hasNativeARC() const {
    switch (getKind()) {
    case FragileMacOSX: return getVersion() >= VersionTuple(10, 7);
    case MacOSX: return getVersion() >= VersionTuple(10, 7);
    case iOS: return getVersion() >= VersionTuple(5);
    case WatchOS: return true;

    case GCC: return false;
    case GNUstep: return getVersion() >= VersionTuple(1, 6);
    case ObjFW: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Does this runtime supports optimized setter entrypoints?
  bool hasOptimizedSetter() const {
    switch (getKind()) {
      case MacOSX:
        return getVersion() >= VersionTuple(10, 8);
      case iOS:
        return (getVersion() >= VersionTuple(6));
      case WatchOS:
        return true;
      case GNUstep:
        return getVersion() >= VersionTuple(1, 7);
    
      default:
      return false;
    }
  }

  /// Does this runtime allow the use of __weak?
  bool allowsWeak() const {
    return hasNativeWeak();
  }

  /// \brief Does this runtime natively provide ARC-compliant 'weak'
  /// entrypoints?
  bool hasNativeWeak() const {
    // Right now, this is always equivalent to whether the runtime
    // natively supports ARC decision.
    return hasNativeARC();
  }

  /// \brief Does this runtime directly support the subscripting methods?
  ///
  /// This is really a property of the library, not the runtime.
  bool hasSubscripting() const {
    switch (getKind()) {
    case FragileMacOSX: return false;
    case MacOSX: return getVersion() >= VersionTuple(10, 8);
    case iOS: return getVersion() >= VersionTuple(6);
    case WatchOS: return true;

    // This is really a lie, because some implementations and versions
    // of the runtime do not support ARC.  Probably -fgnu-runtime
    // should imply a "maximal" runtime or something?
    case GCC: return true;
    case GNUstep: return true;
    case ObjFW: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Does this runtime allow sizeof or alignof on object types?
  bool allowsSizeofAlignof() const {
    return isFragile();
  }

  /// \brief Does this runtime allow pointer arithmetic on objects?
  ///
  /// This covers +, -, ++, --, and (if isSubscriptPointerArithmetic()
  /// yields true) [].
  bool allowsPointerArithmetic() const {
    switch (getKind()) {
    case FragileMacOSX:
    case GCC:
      return true;
    case MacOSX:
    case iOS:
    case WatchOS:
    case GNUstep:
    case ObjFW:
      return false;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Is subscripting pointer arithmetic?
  bool isSubscriptPointerArithmetic() const {
    return allowsPointerArithmetic();
  }

  /// \brief Does this runtime provide an objc_terminate function?
  ///
  /// This is used in handlers for exceptions during the unwind process;
  /// without it, abort() must be used in pure ObjC files.
  bool hasTerminate() const {
    switch (getKind()) {
    case FragileMacOSX: return getVersion() >= VersionTuple(10, 8);
    case MacOSX: return getVersion() >= VersionTuple(10, 8);
    case iOS: return getVersion() >= VersionTuple(5);
    case WatchOS: return true;
    case GCC: return false;
    case GNUstep: return false;
    case ObjFW: return false;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Does this runtime support weakly importing classes?
  bool hasWeakClassImport() const {
    switch (getKind()) {
    case MacOSX: return true;
    case iOS: return true;
    case WatchOS: return true;
    case FragileMacOSX: return false;
    case GCC: return true;
    case GNUstep: return true;
    case ObjFW: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// \brief Does this runtime use zero-cost exceptions?
  bool hasUnwindExceptions() const {
    switch (getKind()) {
    case MacOSX: return true;
    case iOS: return true;
    case WatchOS: return true;
    case FragileMacOSX: return false;
    case GCC: return true;
    case GNUstep: return true;
    case ObjFW: return true;
    }
    llvm_unreachable("bad kind");
  }

  bool hasAtomicCopyHelper() const {
    switch (getKind()) {
    case FragileMacOSX:
    case MacOSX:
    case iOS:
    case WatchOS:
      return true;
    case GNUstep:
      return getVersion() >= VersionTuple(1, 7);
    default: return false;
    }
  }

  /// Is objc_unsafeClaimAutoreleasedReturnValue available?
  bool hasARCUnsafeClaimAutoreleasedReturnValue() const {
    switch (getKind()) {
    case MacOSX:
    case FragileMacOSX:
      return getVersion() >= VersionTuple(10, 11);
    case iOS:
      return getVersion() >= VersionTuple(9);
    case WatchOS:
      return getVersion() >= VersionTuple(2);
    case GNUstep:
      return false;

    default:
      return false;
    }
  }

  /// \brief Try to parse an Objective-C runtime specification from the given
  /// string.
  ///
  /// \return true on error.
  bool tryParse(StringRef input);

  std::string getAsString() const;

  friend bool operator==(const ObjCRuntime &left, const ObjCRuntime &right) {
    return left.getKind() == right.getKind() &&
           left.getVersion() == right.getVersion();
  }

  friend bool operator!=(const ObjCRuntime &left, const ObjCRuntime &right) {
    return !(left == right);
  }
};

raw_ostream &operator<<(raw_ostream &out, const ObjCRuntime &value);

}  // end namespace clang

#endif
