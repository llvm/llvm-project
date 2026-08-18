// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void user_func(int *__restrict p);

void test_user(int *__restrict p) {
  user_func(p);
}

// CIR: cir.func private @user_func(!cir.ptr<!s32i> {llvm.noalias, llvm.noundef})
// CIR: cir.func {{.*}} @test_user(%arg0: !cir.ptr<!s32i> {llvm.noalias, llvm.noundef}
// CIR:   cir.call @user_func(%{{.*}}) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()

// LLVM: define dso_local void @test_user(ptr noalias noundef %{{.*}})
// LLVM:   call void @user_func(ptr noundef %{{.*}})

// OGCG: define dso_local void @test_user(ptr noalias noundef %{{.*}})
// OGCG:   call void @user_func(ptr noundef %{{.*}})

int printf(const char *__restrict fmt, ...);

void test_builtin(const char *__restrict fmt) {
  printf(fmt);
}

// Builtins must NOT get noalias from restrict (matching OGCG behavior).
// CIR: cir.func {{.*}} @test_builtin(%arg0: !cir.ptr<!s8i> {llvm.noalias, llvm.noundef}
// CIR:   cir.call @printf(%{{.*}}) : (!cir.ptr<!s8i> {llvm.noundef}) -> !s32i

// LLVM: define dso_local void @test_builtin(ptr noalias noundef %{{.*}})
// LLVM:   call i32 (ptr, ...) @printf(ptr noundef %{{.*}})

// OGCG: define dso_local void @test_builtin(ptr noalias noundef %{{.*}})
// OGCG:   call i32 (ptr, ...) @printf(ptr noundef %{{.*}})
