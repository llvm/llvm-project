// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -debug-info-kind=limited -emit-llvm -o - | FileCheck -check-prefix=CHECK -check-prefix=DARWIN-X64 %s
// RUN: %clang_cc1 %s -triple x86_64-pc-win32     -debug-info-kind=limited -emit-llvm -o - | FileCheck -check-prefix=CHECK -check-prefix=WIN32-X64 %s

struct T {
  int method();
};

void foo(int (T::*method)()) {}

// A pointer to a member function is a pair of function- and this-pointer.
// CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type,
// DARWIN-X64-SAME:      size: 128
// WIN32-X64-SAME:       size: 64

struct Incomplete;

int (Incomplete::**bar)();
// CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type,
// DARWIN-X64-SAME:           size: 128
// WIN32-X64-NOT:             size:
// CHECK-SAME:                extraData: {{.*}})
