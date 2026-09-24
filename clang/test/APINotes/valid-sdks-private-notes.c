// RUN: rm -rf %t

// Both API notes apply on macOS 10.15.
// RUN: %clang_cc1 -triple arm64-apple-macosx10.15 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/both -fapinotes-modules \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX10.15.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=both

// The public API notes are skipped on macOS 13.0, but the private API notes should
// still apply.
// RUN: %clang_cc1 -triple arm64-apple-macosx13.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/private-only -fapinotes-modules \
// RUN:   -isysroot %S/../InstallAPI/Inputs/MacOSX13.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=privateonly

// The private API notes are skipped on iOS 13.0, but the public API notes should
// still apply.
// RUN: %clang_cc1 -triple arm64-apple-ios13.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/public-only -fapinotes-modules \
// RUN:   -isysroot %S/../Driver/Inputs/iPhoneOS13.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=publiconly

// Both API notes are skipped on macOS 15.0.
// RUN: %clang_cc1 -triple arm64-apple-macosx15.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/neither -fapinotes-modules \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=neither

// neither-no-diagnostics

#include "PublicPrivateLib.h"

void test(void) {
  public_note_fn();
  // both-error@-1{{'public_note_fn' is unavailable: from the public API notes}}
  // both-note@PublicPrivateLib.h:5{{'public_note_fn' has been explicitly marked unavailable here}}
  // publiconly-error@-3{{'public_note_fn' is unavailable: from the public API notes}}
  // publiconly-note@PublicPrivateLib.h:5{{'public_note_fn' has been explicitly marked unavailable here}}

  private_note_fn();
  // both-error@-1{{'private_note_fn' is unavailable: from the private API notes}}
  // both-note@PublicPrivateLib.h:8{{'private_note_fn' has been explicitly marked unavailable here}}
  // privateonly-error@-3{{'private_note_fn' is unavailable: from the private API notes}}
  // privateonly-note@PublicPrivateLib.h:8{{'private_note_fn' has been explicitly marked unavailable here}}
}
