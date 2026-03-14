// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s

// Triggers the deserialization of B's destructor.
B b1;

// CHECK: CXXDestructorDecl

// CHECK-NEXT: ~B 'void () noexcept' virtual
// CHECK-SAME: 'void () noexcept'
// CHECK-SAME: virtual
