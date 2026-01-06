// Check cas-fs-based caching works with availability check based on
// SDKSettings.json.

// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   -cas-path %t/cas > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=mod > %t/mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: not %clang @%t/mod.rsp 2>&1 | FileCheck %s

// CHECK: error: 'fUnavail' is unavailable

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -target x86_64-apple-macos11 -fsyntax-only DIR/tu.c -isysroot DIR/MacOSX11.0.sdk -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps",
  "file": "DIR/tu.c"
}]

//--- MacOSX11.0.sdk/SDKSettings.json
{
  "DefaultVariant": "macos", "DisplayName": "macOS 11",
  "CanonicalName": "macosx11.0", "Version": "11.0",
  "MaximumDeploymentTarget": "11.0.99",
  "SupportedTargets": {
    "macosx": {
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"],
      "LLVMTargetTripleVendor": "apple",
      "LLVMTargetTripleSys": "macosx",
      "LLVMTargetTripleEnvironment": ""
    },
    "iosmac": {
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"],
      "LLVMTargetTripleVendor": "apple",
      "LLVMTargetTripleSys": "ios",
      "LLVMTargetTripleEnvironment": "macabi"
    }
  }
}

//--- module.modulemap
module mod { header "mod.h" }

//--- mod.h
void fUnavail(void) __attribute__((availability(macOS, obsoleted = 10.15)));

static inline void module(void) {
  fUnavail();
}

//--- tu.c
#include "mod.h"
