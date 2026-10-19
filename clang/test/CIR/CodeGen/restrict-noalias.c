// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-ignored-attributes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-ignored-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-ignored-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void user_func(int *__restrict p);

void test_user(int *__restrict p) {
  user_func(p);
}

// CIR: cir.func {{.*}} @test_user(%arg0: !cir.ptr<!s32i> {llvm.noalias, llvm.noundef}
// CIR:   cir.call @user_func(%{{.*}}) : (!cir.ptr<!s32i> {llvm.noundef}) -> ()
// CIR: cir.func private @user_func(!cir.ptr<!s32i> {llvm.noalias, llvm.noundef})

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

__attribute__((malloc)) void *my_malloc(unsigned long n);
void *test_ret(unsigned long n) { return my_malloc(n); }

// CIR: cir.func {{.*}} @test_ret(%{{.*}}: !u64i {llvm.noundef}
// CIR:   cir.call @my_malloc(%{{.*}}) : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.noalias{{.*}}})
// CIR: cir.func {{.*}} @my_malloc(!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.noalias{{.*}}})

// LLVM: define dso_local {{.*}}ptr @test_ret
// LLVM:   call noalias {{.*}}ptr @my_malloc
// LLVM: declare noalias {{.*}}ptr @my_malloc

// OGCG: define dso_local {{.*}}ptr @test_ret
// OGCG:   call noalias {{.*}}ptr @my_malloc
// OGCG: declare noalias {{.*}}ptr @my_malloc

__attribute__((malloc)) void *my_malloc_def(unsigned long n) { return 0; }

// CIR: cir.func {{.*}} @my_malloc_def(%{{.*}}: !u64i {llvm.noundef}
// CIR-SAME: -> (!cir.ptr<!void> {llvm.noalias{{.*}}})

// LLVM: define dso_local noalias {{.*}}ptr @my_malloc_def
// OGCG: define dso_local noalias {{.*}}ptr @my_malloc_def

int *Mem;
void dealloc(int *);
__attribute__((malloc(dealloc))) int *malloc_with_dealloc(void) { return Mem; }
__attribute__((malloc(dealloc, 1))) int *malloc_with_dealloc_idx(void) {
  return Mem;
}

int *test_malloc_with_dealloc(void) { return malloc_with_dealloc(); }

// CIR-LABEL: cir.func {{.*}} @malloc_with_dealloc
// CIR-NOT: llvm.noalias
// CIR-LABEL: cir.func {{.*}} @malloc_with_dealloc_idx
// CIR-NOT: llvm.noalias
// CIR-LABEL: cir.func {{.*}} @test_malloc_with_dealloc
// CIR: cir.call @malloc_with_dealloc() : () -> !cir.ptr<!s32i>
// CIR-NOT: llvm.noalias

// Bracket the return-attribute region. An unrestricted {{.*}} before ptr would
// let FileCheck consume an incorrect noalias and still satisfy the -NOT.
// LLVM: define dso_local
// LLVM-NOT: noalias
// LLVM-SAME: ptr @malloc_with_dealloc()
// LLVM: define dso_local
// LLVM-NOT: noalias
// LLVM-SAME: ptr @malloc_with_dealloc_idx()
// LLVM: define {{.*}} @test_malloc_with_dealloc()
// LLVM: call
// LLVM-NOT: noalias
// LLVM-SAME: ptr @malloc_with_dealloc()

// OGCG: define dso_local
// OGCG-NOT: noalias
// OGCG-SAME: ptr @malloc_with_dealloc()
// OGCG: define dso_local
// OGCG-NOT: noalias
// OGCG-SAME: ptr @malloc_with_dealloc_idx()
// OGCG: define {{.*}} @test_malloc_with_dealloc()
// OGCG: call
// OGCG-NOT: noalias
// OGCG-SAME: ptr @malloc_with_dealloc()
