// Check that darwin passes -fbuiltin-headers-in-system-modules
// when expected.

// RUN: %clang -target x86_64-apple-darwin22.4 -### %s 2>&1 | FileCheck %s
// RUN: %clang -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -target x86_64-apple-macos10.15 -### %s 2>&1 | FileCheck %s
// RUN: %clang -isysroot %S/Inputs/iPhoneOS13.0.sdk -target arm64-apple-ios13.0 -### %s 2>&1 | FileCheck %s
// CHECK: -fbuiltin-headers-in-system-modules

// RUN: %clang -isysroot %S/Inputs/MacOSX99.0.sdk -target x86_64-apple-macos98.0 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/MacOSX99.0.sdk -target x86_64-apple-macos99.0 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// CHECK_FUTURE-NOT: -fbuiltin-headers-in-system-modules


// Check that builtin_headers_in_system_modules is only set if -fbuiltin-headers-in-system-modules and -fmodules are both set.

// RUN: %clang -isysroot %S/Inputs/iPhoneOS13.0.sdk -target arm64-apple-ios13.0 -fsyntax-only %s -Xclang -verify=no-feature
// RUN: %clang -isysroot %S/Inputs/iPhoneOS13.0.sdk -target arm64-apple-ios13.0 -fsyntax-only %s -fmodules -Xclang -verify=yes-feature
// RUN: %clang -isysroot %S/Inputs/MacOSX99.0.sdk -target x86_64-apple-macos99.0 -fsyntax-only %s -Xclang -verify=no-feature
// RUN: %clang -isysroot %S/Inputs/MacOSX99.0.sdk -target x86_64-apple-macos99.0 -fsyntax-only %s -fmodules -Xclang -verify=no-feature

#if __has_feature(builtin_headers_in_system_modules)
#error "has builtin_headers_in_system_modules"
// yes-feature-error@-1 {{}}
#else
#error "no builtin_headers_in_system_modules"
// no-feature-error@-1 {{}}
#endif
