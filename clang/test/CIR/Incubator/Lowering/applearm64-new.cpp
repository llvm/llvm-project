// RUN: %clang_cc1 -triple=arm64e-apple-darwin -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

class C {
  public:
    ~C();
};

void t_constant_size_nontrivial() {
  auto p = new C[3];
}

// Note: The below differs from the IR emitted by clang without -fclangir in
//       several respects. (1) The alloca here has an extra "i64 1"
//       (2) The operator new call is missing "noalias noundef nonnull" on
//       the call and "noundef" on the argument, (3) The getelementptr is
//       missing "inbounds"

// LLVM: @_Z26t_constant_size_nontrivialv()
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 19)
// LLVM:   store i64 1, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[NUM_ELEMENTS_PTR:.*]] = getelementptr i64, ptr %[[COOKIE_PTR]], i64 1
// LLVM:   store i64 3, ptr %[[NUM_ELEMENTS_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 16

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// LLVM: @_Z27t_constant_size_nontrivial2v()
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 28)
// LLVM:   store i64 4, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[NUM_ELEMENTS_PTR:.*]] = getelementptr i64, ptr %[[COOKIE_PTR]], i64 1
// LLVM:   store i64 3, ptr %[[NUM_ELEMENTS_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 16
