// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void local_typedef(void) {
  typedef struct {int a;} Struct;
  Struct s;
}

// CIR:      cir.func @local_typedef()
// CIR:        cir.alloca !rec_Struct, !cir.ptr<!rec_Struct>, ["s"] {alignment = 4 : i64}
// CIR:        cir.return

// LLVM: %struct.Struct = type { i32 }
// LLVM: define void @local_typedef()
// LLVM:   alloca %struct.Struct, i64 1, align 4
// LLVM:   ret void

// OGCG: %struct.Struct = type { i32 }
// OGCG: define{{.*}} void @local_typedef()
// OGCG:   alloca %struct.Struct, align 4
// OGCG:   ret void
