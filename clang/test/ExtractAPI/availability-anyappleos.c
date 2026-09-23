// RUN: rm -rf %t

/*
 * This test file looks slightly different from the other ones since the
 * `anyAppleOS` availability attribute needs to be tested for multiple
 * target platforms. To make the tests readable, the commands to build
 * the symbol graphs and check the contained availability info have been
 * grouped per-platform below.
 */

void anyappleos_introduced(void) __attribute__((availability(anyAppleOS, introduced=26.0)));

void anyappleos_all(void) __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0, obsoleted=28.0)));

void anyappleos_unavailable(void) __attribute__((availability(anyAppleOS, unavailable)));

void anyappleos_explicit_override(void)
    __attribute__((availability(anyAppleOS, introduced=26.0)))
    __attribute__((availability(macOS, introduced=27.0)));

// expected-no-diagnostics

// ========== macOS ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-macosx \
// RUN:   -x c-header %s -o %t/macos.symbols.json -verify
// RUN: FileCheck %s --input-file %t/macos.symbols.json --check-prefix MACOS

// MACOS-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// MACOS:      "availability": [
// MACOS-NEXT:   {
// MACOS-NEXT:     "domain": "macos",
// MACOS-NEXT:     "introduced": {
// MACOS-NEXT:       "major": 26,
// MACOS-NEXT:       "minor": 0,
// MACOS-NEXT:       "patch": 0
// MACOS-NEXT:     }
// MACOS-NEXT:   }
// MACOS-NEXT: ]

// MACOS-LABEL: "!testLabel": "c:@F@anyappleos_all"
// MACOS:      "availability": [
// MACOS-NEXT:   {
// MACOS-NEXT:     "deprecated": {
// MACOS-NEXT:       "major": 27,
// MACOS-NEXT:       "minor": 0,
// MACOS-NEXT:       "patch": 0
// MACOS-NEXT:     },
// MACOS-NEXT:     "domain": "macos",
// MACOS-NEXT:     "introduced": {
// MACOS-NEXT:       "major": 26,
// MACOS-NEXT:       "minor": 0,
// MACOS-NEXT:       "patch": 0
// MACOS-NEXT:     },
// MACOS-NEXT:     "obsoleted": {
// MACOS-NEXT:       "major": 28,
// MACOS-NEXT:       "minor": 0,
// MACOS-NEXT:       "patch": 0
// MACOS-NEXT:     }
// MACOS-NEXT:   }
// MACOS-NEXT: ]

// MACOS-LABEL: "!testLabel": "c:@F@anyappleos_unavailable"
// MACOS:      "availability": [
// MACOS-NEXT:   {
// MACOS-NEXT:     "domain": "macos",
// MACOS-NEXT:     "isUnconditionallyUnavailable": true
// MACOS-NEXT:   }
// MACOS-NEXT: ]

// MACOS-LABEL: "!testLabel": "c:@F@anyappleos_explicit_override"
// MACOS:      "availability": [
// MACOS-NEXT:   {
// MACOS-NEXT:     "domain": "macos",
// MACOS-NEXT:     "introduced": {
// MACOS-NEXT:       "major": 27,
// MACOS-NEXT:       "minor": 0,
// MACOS-NEXT:       "patch": 0
// MACOS-NEXT:     }
// MACOS-NEXT:   }
// MACOS-NEXT: ]

// ========== iOS ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-ios \
// RUN:   -x c-header %s -o %t/ios.symbols.json -verify
// RUN: FileCheck %s --input-file %t/ios.symbols.json --check-prefix IOS

// IOS-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// IOS:      "availability": [
// IOS-NEXT:   {
// IOS-NEXT:     "domain": "ios",
// IOS-NEXT:     "introduced": {
// IOS-NEXT:       "major": 26,
// IOS-NEXT:       "minor": 0,
// IOS-NEXT:       "patch": 0
// IOS-NEXT:     }
// IOS-NEXT:   }
// IOS-NEXT: ]

// IOS-LABEL: "!testLabel": "c:@F@anyappleos_unavailable"
// IOS:      "availability": [
// IOS-NEXT:   {
// IOS-NEXT:     "domain": "ios",
// IOS-NEXT:     "isUnconditionallyUnavailable": true
// IOS-NEXT:   }
// IOS-NEXT: ]

// IOS-LABEL: "!testLabel": "c:@F@anyappleos_explicit_override"
// IOS:      "availability": [
// IOS-NEXT:   {
// IOS-NEXT:     "domain": "ios",
// IOS-NEXT:     "introduced": {
// IOS-NEXT:       "major": 26,
// IOS-NEXT:       "minor": 0,
// IOS-NEXT:       "patch": 0
// IOS-NEXT:     }
// IOS-NEXT:   }
// IOS-NEXT: ]

// ========== tvOS ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-tvos \
// RUN:   -x c-header %s -o %t/tvos.symbols.json -verify
// RUN: FileCheck %s --input-file %t/tvos.symbols.json --check-prefix TVOS

// TVOS-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// TVOS:       "domain": "tvos"

// ========== watchOS ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-watchos \
// RUN:   -x c-header %s -o %t/watchos.symbols.json -verify
// RUN: FileCheck %s --input-file %t/watchos.symbols.json --check-prefix WATCHOS

// WATCHOS-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// WATCHOS:       "domain": "watchos"

// ========== xrOS ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-xros \
// RUN:   -x c-header %s -o %t/xros.symbols.json -verify
// RUN: FileCheck %s --input-file %t/xros.symbols.json --check-prefix XROS

// XROS-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// XROS:       "domain": "xros"

// ========== macCatalyst ==========
/*
 * The macCatalyst run does not use -verify because an explicit
 * `availability(macos, ...)` attribute on a macabi triple emits an
 * SDKSettings.json warning which is irrelevant to the test.
 */
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-ios-macabi \
// RUN:   -x c-header %s -o %t/maccatalyst.symbols.json
// RUN: FileCheck %s --input-file %t/maccatalyst.symbols.json --check-prefix MACCATALYST

// MACCATALYST-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// MACCATALYST:       "domain": "maccatalyst"

// ========== driverkit ==========
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -triple arm64-apple-driverkit \
// RUN:   -x c-header %s -o %t/driverkit.symbols.json -verify
// RUN: FileCheck %s --input-file %t/driverkit.symbols.json --check-prefix DRIVERKIT

// DRIVERKIT-LABEL: "!testLabel": "c:@F@anyappleos_introduced"
// DRIVERKIT:       "domain": "driverkit"
