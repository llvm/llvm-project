// RUN: rm -rf %t

// API notes without ValidSDKs/ValidUntil always apply.
// RUN: %clang_cc1 -triple arm64-apple-macosx15.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/no-key -fapinotes-modules \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=nokey

// Old SDK versions get the API notes, even when a new delopment version is used.
// RUN: %clang_cc1 -triple arm64-apple-macosx15.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/below -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX10.15.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// API notes are skipped when the SDK reaches the ValidUntil version.
// RUN: %clang_cc1 -triple arm64-apple-macosx10.15 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/at-boundary -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=skipped

// New SDK versions skip the API notes, even when an old deployment version is used.
// RUN: %clang_cc1 -triple arm64-apple-ios11.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/newer-than-cutoff \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/iPhoneOS13.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=skipped

// Totally new SDKs skip the API notes.
// RUN: %clang_cc1 -triple arm64-apple-xros1.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/unlisted-platform \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/XROS1.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=skipped

// Particularly old SDK versions have no SDKSettings, the API notes apply even if the
// SDK wasn't listed.
// RUN: %clang_cc1 -triple x86_64-apple-driverkit19.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/no-sdksettings \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/DriverKit19.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// An -isysroot pointing inside the SDK is a gray area, err on the side of applying
// the API notes.
// RUN: %clang_cc1 -triple arm64-apple-xros1.0-simulator -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/nested-sysroot \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -isysroot %S/../Driver/Inputs/XRSimulator1.0.sdk/usr/include/libxml \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// No SDK provided, the triple doesn't get used as a fallback.
// RUN: %clang_cc1 -triple arm64-apple-macosx15.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/no-sdk -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// Non-Darwin targets have nothing to fall back on, API notes always apply.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/non-darwin -iapinotes-modules %S/Inputs/ValidSDKs/multiple-sdks \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// Device and simulator SDKs have to both be specified.
// RUN: %clang_cc1 -triple arm64-apple-xros1.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/device -iapinotes-modules %S/Inputs/ValidSDKs/device-only \
// RUN:   -isysroot %S/../Driver/Inputs/XROS1.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies
// RUN: %clang_cc1 -triple arm64-apple-xros1.0-simulator -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/simulator -iapinotes-modules %S/Inputs/ValidSDKs/device-only \
// RUN:   -isysroot %S/../Driver/Inputs/XRSimulator1.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=skipped

// Device and simulator SDKs can have different versions.
// RUN: %clang_cc1 -triple arm64-apple-xros1.0 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/device-versioned \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/different-simulator-version \
// RUN:   -isysroot %S/../Driver/Inputs/XROS1.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=skipped
// RUN: %clang_cc1 -triple arm64-apple-xros2.0-simulator -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/simulator-versioned \
// RUN:   -iapinotes-modules %S/Inputs/ValidSDKs/different-simulator-version \
// RUN:   -isysroot %S/../Driver/Inputs/XRSimulator1.0.sdk \
// RUN:   -fsyntax-only -I %S/Inputs/Headers %s -verify=applies

// skipped-no-diagnostics

#include "AgingLib.h"
// HeaderLib's existing API notes declare no ValidSDKs; only the first RUN above
// enables them, so the other cases see this declaration unannotated.
#include "HeaderLib.h"

void test(void) {
  aged_out_function();
  // applies-error@-1{{'aged_out_function' is unavailable: the SDK now declares this itself}}
  // applies-note@AgingLib.h:6{{'aged_out_function' has been explicitly marked unavailable here}}

  unavailable_function();
  // nokey-error@-1{{'unavailable_function' is unavailable: I beg you not to use this}}
  // nokey-note@HeaderLib.h:8{{'unavailable_function' has been explicitly marked unavailable here}}
}
