// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

extern int x;
void g(void);

// The call to g() is unreachable, but we currently want to convert it during
// the CIR-to-LLVM lowering pass to avoid verification errors.

void f(void) {
  goto end;
lab:
  if (0) goto lab;
  if (x) g();
end:;
}

// CIR-DAG: cir.global "private" external{{.*}} @x
// CIR:     cir.func{{.*}} @f
// CIR:       cir.goto "end"
// CIR:       cir.label "lab"
// CIR:       cir.get_global @x

// LLVM:     @x = external {{(dso_local )?}}global i32
// LLVM:     define dso_local void @f
// LLVM:       load i32, ptr @x

// OGCG:     @x = external {{.*}}global i32
// OGCG:     define dso_local void @f
// OGCG:       load i32, ptr @x
