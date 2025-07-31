// This test checks that the module cache gets invalidated when the SDKSettings.json file changes.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- AppleTVOS15.0.sdk/SDKSettings-old.json
{
  "DisplayName": "tvOS 15.0",
  "Version": "15.0",
  "CanonicalName": "appletvos15.0",
  "MaximumDeploymentTarget": "15.0.99",
  "PropertyConditionFallbackNames": []
}
//--- AppleTVOS15.0.sdk/SDKSettings-new.json
{
  "DisplayName": "tvOS 15.0",
  "Version": "15.0",
  "CanonicalName": "appletvos15.0",
  "MaximumDeploymentTarget": "15.0.99",
  "PropertyConditionFallbackNames": [],
  "VersionMap": {
    "iOS_tvOS": {
      "13.2": "13.1"
    },
    "tvOS_iOS": {
      "13.1": "13.2"
    }
  }
}
//--- module.modulemap
module M { header "M.h" }
//--- M.h
void foo(void) __attribute__((availability(iOS, obsoleted = 13.2)));
void test() { foo(); }

//--- tu.m
#include "M.h"

// Compiling for tvOS 13.1 without "VersionMap" should succeed, since by default iOS 13.2 gets mapped to tvOS 13.2,
// and \c foo is therefore **not** deprecated.
// RUN: cp %t/AppleTVOS15.0.sdk/SDKSettings-old.json %t/AppleTVOS15.0.sdk/SDKSettings.json
// RUN: %clang -target x86_64-apple-tvos13.1 -isysroot %t/AppleTVOS15.0.sdk \
// RUN:   -fsyntax-only %t/tu.m -o %t/tu.o -fmodules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/cache

// Compiling for tvOS 13.1 with "VersionMap" saying it maps to iOS 13.2 should fail, since \c foo is now deprecated.
// RUN: sleep 1
// RUN: cp %t/AppleTVOS15.0.sdk/SDKSettings-new.json %t/AppleTVOS15.0.sdk/SDKSettings.json
// RUN: not %clang -target x86_64-apple-tvos13.1 -isysroot %t/AppleTVOS15.0.sdk \
// RUN:   -fsyntax-only %t/tu.m -o %t/tu.o -fmodules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/cache 2>&1 \
// RUN:     | FileCheck %s
// CHECK: M.h:2:15: error: 'foo' is unavailable: obsoleted in tvOS 13.1
// CHECK: M.h:1:6: note: 'foo' has been explicitly marked unavailable here
// CHECK: tu.m:1:10: fatal error: could not build module 'M'
