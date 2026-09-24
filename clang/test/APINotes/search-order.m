// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_FRAMEWORK=1 -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %S/Inputs/APINotes  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES=1 -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -iapinotes-modules %S/Inputs/APINotes  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_FRAMEWORK=1 -verify

// First search path wins.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/APINotesFirst -iapinotes-modules %S/Inputs/APINotes -iapinotes-modules %S/Inputs/APINotesAged -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES=1 -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/APINotesAgedFirst -iapinotes-modules %S/Inputs/APINotesAged -iapinotes-modules %S/Inputs/APINotes -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES_AGED=1 -verify

// Use -isysroot to selectively skip the API notes which use ValidSDKs/ValidUntil.
// Skipped API notes should behave as if they weren't present at all. If non-skipped
// API notes are found, they should be used even if skipped ones are later in the
// search path. If skipped API notes are found, the search should continue through
// the search path for non-skipped API notes.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModuleNotesAgedOut -fapinotes-modules -iapinotes-modules %S/Inputs/APINotes -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES=1 -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/SearchPathNotConsulted -fapinotes-modules -iapinotes-modules %S/Inputs/APINotesAged -isysroot %S/../Driver/Inputs/MacOSX10.15.sdk -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_FRAMEWORK=1 -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/APINotesAgedFirstWithSDK -iapinotes-modules %S/Inputs/APINotesAged -iapinotes-modules %S/Inputs/APINotes -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES=1 -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/APINotesFirstWithSDK -iapinotes-modules %S/Inputs/APINotes -iapinotes-modules %S/Inputs/APINotesAged -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_APINOTES=1 -verify

@import SomeOtherKit;

void test(A *a) {
#if FROM_FRAMEWORK
  [a methodA]; // expected-error{{unavailable}}
  [a methodB];
  [a methodC];

  // expected-note@SomeOtherKit/SomeOtherKit.h:5{{'methodA' has been explicitly marked unavailable here}}
#elif FROM_APINOTES
  [a methodA];
  [a methodB]; // expected-error{{unavailable}}
  [a methodC];

  // expected-note@SomeOtherKit/SomeOtherKit.h:6{{'methodB' has been explicitly marked unavailable here}}
#elif FROM_APINOTES_AGED
  [a methodA];
  [a methodB];
  [a methodC]; // expected-error{{unavailable}}

  // expected-note@SomeOtherKit/SomeOtherKit.h:7{{'methodC' has been explicitly marked unavailable here}}
#else
#  error Not something we need to test
#endif
}
