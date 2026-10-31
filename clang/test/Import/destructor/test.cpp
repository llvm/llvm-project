// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// Triggers the deserialization of B's destructor.
B b1;

// CHECK:      CXXDestructorDecl
// CHECK-SAME: ~B
// CHECK-SAME: 'void () noexcept'
// CHECK-SAME: virtual
