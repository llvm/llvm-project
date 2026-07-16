// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Delayed;
Delayed (*fp)();

struct Delayed {
  int x;
};

void use() {
  Delayed result = fp();
}

// CIR: cir.global external @fp = #cir.ptr<null> : !cir.ptr<!cir.func<() -> !rec_Delayed>>

// CIR: cir.func {{.*}} @_Z3usev()
// CIR:   %[[FP:.*]] = cir.get_global @fp
// CIR:   %[[LOAD:.*]] = cir.load {{.*}} %[[FP]]
// CIR:   cir.call %[[LOAD]]() : (!cir.ptr<!cir.func<() -> !rec_Delayed>>) -> !rec_Delayed

// The difference between LLVM and OGCG is due to missing ABI lowering.

// LLVM: @fp = global ptr null
// LLVM: define {{.*}} void @_Z3usev()
// LLVM:   %[[FP:.*]] = load ptr, ptr @fp
// LLVM:   %[[CALL:.*]] = call %struct.Delayed %[[FP]]()

// OGCG: @fp = global ptr null
// OGCG: define {{.*}} void @_Z3usev()
// OGCG:   %[[FP:.*]] = load ptr, ptr @fp
// OGCG:   call i32 %[[FP]]()
