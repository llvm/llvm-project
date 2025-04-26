// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `\`.

// Add default directories before running clang to check default 
// search paths. 
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp -R %S/Inputs/MacOSX15.1.sdk %t/
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/Frameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/SubFrameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/include

// RUN: %clang %s -target arm64-apple-darwin13.0 -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck %s

// CHECK:    -isysroot [[PATH:[^ ]*/MacOSX15.1.sdk]]
// CHECK:    #include <...> search starts here:
// CHECK:    [[PATH]]/usr/include
// CHECK:    [[PATH]]/System/Library/Frameworks (framework directory)
// CHECK:    [[PATH]]/System/Library/SubFrameworks (framework directory)
