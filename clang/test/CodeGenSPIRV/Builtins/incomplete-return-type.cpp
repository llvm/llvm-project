// Regression test for de82b4790943: calling hasTrivialCopyConstructor() on an
// incomplete CXXRecordDecl in withReturnValueSlot (CGExprAgg.cpp) must not
// assert. The CustomTypeChecking builtin bypasses CheckCallReturnType in Sema,
// allowing an incomplete class type to reach CodeGen.
//
// With the fix, CodeGen proceeds past withReturnValueSlot and crashes later in
// getASTRecordLayout (which cannot compute layout for forward declarations).
// We verify the crash is NOT the CXXRecordDecl::data() assertion.
//
// REQUIRES: asserts
// RUN: not --crash %clang_cc1 -triple spirv64 -fsycl-is-device -x c++ \
// RUN:   -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
// CHECK-NOT: queried property of class with no definition
// CHECK: Cannot get layout of forward declarations

class Incomplete;

[[clang::sycl_external]]
__attribute__((clang_builtin_alias(__builtin_spirv_test_incomplete_return)))
Incomplete make_incomplete();

[[clang::sycl_external]] void caller() {
  make_incomplete();
}
