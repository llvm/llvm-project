// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int a;
  double b;
  char c;
};

int test_structured_binding_size() {
  return __builtin_structured_binding_size(S);
}

// CIR: cir.func {{.*}} @_Z28test_structured_binding_sizev()
// CIR:   %[[SIZE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:   cir.store %[[SIZE:.*]], %[[RETVAL:.*]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL:.*]]
// CIR:   cir.return %[[RET:.*]] : !s32i

// LLVM: define{{.*}} i32 @_Z28test_structured_binding_sizev()
// LLVM:   store i32 3, ptr %[[RETVAL:.*]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL:.*]]
// LLVM:   ret i32 %[[RET:.*]]

// OGCG: define{{.*}} i32 @_Z28test_structured_binding_sizev()
// OGCG:   ret i32 3
