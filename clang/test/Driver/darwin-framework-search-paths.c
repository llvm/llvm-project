// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `/` in the test.

// Add default directories before running clang to check default
// search paths.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp -R %S/Inputs/MacOSX15.1.sdk %t/
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/Frameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/SubFrameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/include

// RUN: %clang -xc %s -target arm64-apple-darwin13.0 -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck --check-prefix=CHECK-C %s
//
// CHECK-C:    -isysroot [[PATH:[^ ]*/MacOSX15.1.sdk]]
// CHECK-C:    #include <...> search starts here:
// CHECK-C:    [[PATH]]/usr/include
// CHECK-C:    [[PATH]]/System/Library/Frameworks (framework directory)
// CHECK-C:    [[PATH]]/System/Library/SubFrameworks (framework directory)

// RUN: %clang -xc++ %s -target arm64-apple-darwin13.0 -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck --check-prefix=CHECK-CXX %s
//
// CHECK-CXX:    -isysroot [[PATH:[^ ]*/MacOSX15.1.sdk]]
// CHECK-CXX:    #include <...> search starts here:
// CHECK-CXX:    [[PATH]]/usr/include
// CHECK-CXX:    [[PATH]]/System/Library/Frameworks (framework directory)
// CHECK-CXX:    [[PATH]]/System/Library/SubFrameworks (framework directory)
