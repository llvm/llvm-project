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
  "CanonicalName": "macosx11.0",
  "Version": "11.0",
  "IsBaseSDK": "YES",
  "DisplayName": "macOS 11.0",
  "MinimalDisplayName": "11.0",
  "SupportedTargets": {
    "macosx": {
      "PlatformFamilyName": "macOS",
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "macosx", "LLVMTargetTripleEnvironment": "",
      "BuildVersionPlatformID": "1",
      "DefaultDeploymentTarget": "11.0",
      "MinimumDeploymentTarget": "10.9", "MaximumDeploymentTarget": "11.0.99",
      "ValidDeploymentTargets": ["10.9", "10.10", "10.11", "10.12", "10.13", "10.14", "10.15", "11.0"]
    },
    "iosmac": {
      "Archs": ["x86_64", "x86_64h", "arm64", "arm64e"], "LLVMTargetTripleVendor": "apple", "LLVMTargetTripleSys": "ios", "LLVMTargetTripleEnvironment": "macabi",
      "BuildVersionPlatformID": "6",
      "DefaultDeploymentTarget": "14.2",
      "MinimumDeploymentTarget": "13.1", "MaximumDeploymentTarget": "14.2.99",
      "ValidDeploymentTargets": ["13.1", "13.2", "13.3", "13.3.1", "13.4", "13.5", "14.2"]
    }
  },
  "VersionMap": {
    "macOS_iOSMac": {"10.15": "13.1", "10.15.1": "13.2", "10.15.2": "13.3", "10.15.3": "13.3.1", "10.15.4": "13.4", "10.15.5": "13.5", "11.0": "14.2"},
    "iOSMac_macOS": {"13.1": "10.15", "13.2": "10.15.1", "13.3": "10.15.2", "13.3.1": "10.15.3", "13.4": "10.15.4", "13.5": "10.15.5", "14.2": "11.0"}
  },
  "DefaultDeploymentTarget": "11.0",
  "MaximumDeploymentTarget": "11.0.99"
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
