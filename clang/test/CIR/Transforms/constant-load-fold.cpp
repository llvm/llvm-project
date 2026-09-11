// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-simplify %s -o %t.cir 2> %t-before-simplify.cir
// RUN: FileCheck --input-file=%t-before-simplify.cir %s -check-prefix=CIR-BEFORE-SIMPLIFY
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

int g();
void use(int);

void h(const int *);

// CIR-BEFORE-SIMPLIFY-LABEL: @_Z18fold_constant_loadv
// CIR-LABEL: @_Z18fold_constant_loadv
// LLVM-LABEL: @_Z18fold_constant_loadv
void fold_constant_load() {
  const int x = g();
  h(&x);
  use(x);

  // CIR-BEFORE-SIMPLIFY: %[[ALLOCA:.+]] = cir.alloca "x" align(4) init const : !cir.ptr<!s32i>
  // CIR-BEFORE-SIMPLIFY: %[[INIT:.+]] = cir.call @_Z1gv()
  // CIR-BEFORE-SIMPLIFY: cir.store align(4) %[[INIT]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR-BEFORE-SIMPLIFY: cir.call @_Z1hPKi(%[[ALLOCA]])
  // CIR-BEFORE-SIMPLIFY: %[[RELOAD:.+]] = cir.load align(4) %[[ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR-BEFORE-SIMPLIFY: cir.call @_Z3usei(%[[RELOAD]])

  // CIR: %[[ALLOCA:.+]] = cir.alloca "x" align(4) init const : !cir.ptr<!s32i>
  // CIR: %[[INIT:.+]] = cir.call @_Z1gv()
  // CIR: cir.store align(4) %[[INIT]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.call @_Z1hPKi(%[[ALLOCA]])
  // CIR: cir.call @_Z3usei(%[[INIT]])

  // LLVM: %[[ALLOCA:.+]] = alloca i32, align 4
  // LLVM: %[[INIT:.+]] = tail call noundef i32 @_Z1gv()
  // LLVM: store i32 %[[INIT]], ptr %[[ALLOCA]], align 4
  // LLVM: call void @_Z1hPKi(ptr noundef nonnull %[[ALLOCA]])
  // LLVM: call void @_Z3usei(i32 noundef %[[INIT]])
}
