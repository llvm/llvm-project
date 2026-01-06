// This test checks that the module cache gets invalidated when the
// SDKSettings.json file changes. This prevents "file changed during build"
// errors when the TU does get rescanned and recompiled.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- sdk/SDKSettings.json
{
  "Version": "11.0",
  "CanonicalName": "macosx11.0", 
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
module M { header "M.h" }
//--- M.h
//--- tu.c
#include "M.h"

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_clean.json \
// RUN:   -- %clang -target x86_64-apple-macos11 -isysroot %t/sdk \
// RUN:      -c %t/tu.c -o %t/tu.o -fmodules -fmodules-cache-path=%t/cache

// RUN: sleep 1
// RUN: echo " " >> %t/sdk/SDKSettings.json
// RUN: echo " " >> %t/tu.c

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -o %t/deps_incremental.json \
// RUN:   -- %clang -target x86_64-apple-macos11 -isysroot %t/sdk \
// RUN:      -c %t/tu.c -o %t/tu.o -fmodules -fmodules-cache-path=%t/cache

// RUN: %deps-to-rsp %t/deps_incremental.json --module-name M > %t/M.rsp
// RUN: %deps-to-rsp %t/deps_incremental.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/M.rsp
// RUN: %clang @%t/tu.rsp
