/// Verify invalid OSVersions are diagnosed.

// RUN: not %clang -target arm64-apple-ios20 -c %s 2>&1 | FileCheck %s --check-prefix=IOS
// IOS: error: invalid version number in '-target arm64-apple-ios20'

// RUN: not %clang -target arm64-apple-watchos20 -c %s 2>&1 | FileCheck %s --check-prefix=WATCHOS
// WATCHOS: error: invalid version number in '-target arm64-apple-watchos20'

// RUN: not %clang -target arm64-apple-macosx19 -c %s 2>&1 | FileCheck %s --check-prefix=MAC
// MAC: error: invalid version number in '-target arm64-apple-macosx19'

// RUN: not %clang -target arm64-apple-ios22-macabi -c %s 2>&1 | FileCheck %s --check-prefix=IOSMAC
// IOSMAC: error: invalid version number in '-target arm64-apple-ios22-macabi'

// RUN: not %clang -target arm64-apple-macosx16 -darwin-target-variant arm64-apple-ios22-macabi  -c %s 2>&1 | FileCheck %s --check-prefix=ZIPPERED
// ZIPPERED: error: invalid version number in 'arm64-apple-ios22-macabi'

// RUN: not %clang -target arm64-apple-visionos5 -c %s 2>&1 | FileCheck %s --check-prefix=VISION
// VISION: error: invalid version number in '-target arm64-apple-visionos5'

// RUN: not %clang -target arm64-apple-tvos21 -c %s 2>&1 | FileCheck %s --check-prefix=TV
// TV: error: invalid version number in '-target arm64-apple-tvos21'
