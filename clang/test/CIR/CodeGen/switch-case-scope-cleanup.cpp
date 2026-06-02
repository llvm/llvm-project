// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct Struk {
  ~Struk();
};

void use(int);

int test(int x) {
  switch (x) {
  case 0:
    return 1;
  case 2:
    Struk s;
    use(x);
    return 2;
  }
  return 3;
}

// CIR: cir.func {{.*}} @_Z4testi
// CIR:         cir.scope {
// CIR:           cir.switch
// CIR:             cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR:               cir.cleanup.scope {
// CIR:                 cir.return
// CIR:               } cleanup normal {
// CIR:                 cir.call{{.*}}@_ZN5StrukD1Ev
// CIR:               }
// CIR:         }
// CIR:         cir.const #cir.int<3> : !s32i
// CIR:         cir.return

// LLVM: define dso_local noundef i32 @_Z4testi(i32 noundef %{{.*}})
// LLVM:   switch i32 %{{.*}}, label %{{.*}} [
// LLVM:     i32 0, label %{{.*}}
// LLVM:     i32 2, label %{{.*}}
// LLVM:   ]
// LLVM:   call void @_Z3usei(i32 noundef %{{.*}})
// LLVM:   call void @_ZN5StrukD1Ev(
// LLVM:   ret i32
