// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void t_new_constant_size() {
  auto p = new double[16];
}

// LLVM: @_Z19t_new_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 128)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// LLVM: @_Z28t_new_multidim_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 192)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

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
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 11)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8

class D {
  public:
    int x;
    ~D();
};

void t_constant_size_nontrivial2() {
  auto p = new D[3];
}

// LLVM: @_Z27t_constant_size_nontrivial2v()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[COOKIE_PTR:.*]] = call ptr @_Znam(i64 20)
// LLVM:   store i64 3, ptr %[[COOKIE_PTR]], align 8
// LLVM:   %[[ALLOCATED_PTR:.*]] = getelementptr i8, ptr %[[COOKIE_PTR]], i64 8
// LLVM:   store ptr %[[ALLOCATED_PTR]], ptr %[[ALLOCA]], align 8
