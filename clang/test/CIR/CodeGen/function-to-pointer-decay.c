// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void f(void);

void f1() {
  (void (*)())f;
}

void f2() {
  (*(void (*)(void))f)();
}

void test_lvalue_cast() {
  (*(void (*)(int))f)(42);
}

// CIR-LABEL:   cir.func{{.*}} @f()
// CIR:         cir.func{{.*}} @f1()
// CIR:         cir.return{{.*}}

// CIR-LABEL:   cir.func{{.*}} @f2()
// CIR:         cir.call @f() : () -> ()

// CIR-LABEL:   cir.func{{.*}} @test_lvalue_cast()
// CIR:         %[[S0:.+]] = {{.*}}@f : !cir.ptr<!cir.func<()>>{{.*}}
// CIR:         %[[S1:.+]] = cir.cast{{.*}}%[[S0]] : !cir.ptr<!cir.func<()>>{{.*}}
// CIR:         %[[S2:.+]] = cir.const #cir.int<42> : !s32i
// CIR:         cir.call %[[S1]](%[[S2]]) : (!cir.ptr<!cir.func<(!s32i)>>, !s32i) -> ()

// LLVM-LABEL:  define{{.*}} void @f1()
// LLVM:        ret void
// LLVM:        define{{.*}} void @f2()
// LLVM:        call void @f()
// LLVM:        define{{.*}} void @test_lvalue_cast()
// LLVM:        call void @f(i32 42)

// OGCG-LABEL:  define{{.*}} void @f1()
// OGCG:        ret void
// OGCG:        define{{.*}} void @f2()
// OGCG:        call void @f()
// OGCG:        define{{.*}} void @test_lvalue_cast()
// OGCG:        call void @f(i32 noundef 42)
