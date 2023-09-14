/// Ensure that -fexperimental-omit-vtable-rtti is only allowed if rtti is
/// disabled.

// RUN: not %clang -c -Xclang -fexperimental-omit-vtable-rtti -frtti %s 2>&1 | FileCheck -check-prefix=ERROR %s
// RUN: not %clang -c -Xclang -fexperimental-omit-vtable-rtti -fno-rtti -frtti %s 2>&1 | FileCheck -check-prefix=ERROR %s

// RUN: %clang -c -Xclang -fexperimental-omit-vtable-rtti -fno-rtti %s 2>&1 | FileCheck -check-prefix=NO-ERROR %s --allow-empty
// RUN: %clang -c -Xclang -fno-experimental-omit-vtable-rtti -frtti %s 2>&1 | FileCheck -check-prefix=NO-ERROR %s --allow-empty
// RUN: %clang -c -Xclang -fexperimental-omit-vtable-rtti -Xclang -fno-experimental-omit-vtable-rtti -frtti %s 2>&1 | FileCheck -check-prefix=NO-ERROR %s --allow-empty

// ERROR: -fexperimental-omit-vtable-rtti call only be used with -fno-rtti
// NO-ERROR-NOT: -fexperimental-omit-vtable-rtti call only be used with -fno-rtti
