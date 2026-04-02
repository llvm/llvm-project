// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

typedef union vec3 {
  struct { double x, y, z; };
  double component[3];
} vec3;

// In C++ mode, this doesn't do zero padding.
extern "C" vec3 ret_vec3() {
  // CIR-LABEL: ret_vec3
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !rec_vec3, !cir.ptr<!rec_vec3>, ["__retval"]
  // CIR: %[[GET_ANON:.*]] = cir.get_member %[[RET_ALLOCA]][0] {name = ""}
  // CIR: %[[GET_X:.*]] = cir.get_member %[[GET_ANON]][0] {name = "x"}
  // CIR: %[[FIVE:.*]] = cir.const #cir.fp<5.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[FIVE]], %[[GET_X]]
  // CIR: %[[GET_Y:.*]] = cir.get_member %[[GET_ANON]][1] {name = "y"}
  // CIR: %[[ZERO:.*]] = cir.const #cir.fp<0.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_Y]]
  // CIR: %[[GET_Z:.*]] = cir.get_member %[[GET_ANON]][2] {name = "z"}
  // CIR: %[[ZERO:.*]] = cir.const #cir.fp<0.{{.*}}> : !cir.double
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_Z]]

  // LLVM-LABEL: ret_vec3
  // OGCG-SAME: ptr{{.*}}sret(%union.vec3){{.*}}%[[RET_ALLOCA:.*]])
  // LLVMCIR: %[[RET_ALLOCA:.*]] = alloca %union.vec3
  // LLVM: %[[GET_X:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 0
  // LLVM: store double 5{{.*}}, ptr %[[GET_X]]
  // LLVM: %[[GET_Y:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 1
  // LLVM: store double 0{{.*}}, ptr %[[GET_Y]]
  // LLVM: %[[GET_Z:.*]] = getelementptr {{.*}}, ptr %[[RET_ALLOCA]], i32 0, i32 2
  // LLVM: store double 0{{.*}}, ptr %[[GET_Z]]
  return (vec3) {{ .x = 5.0 }};
}

typedef union Trivial {
  int a;
} Trivial;

extern "C" Trivial ret_trivial() { return {}; }
  // CIR-LABEL: ret_trivial
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !rec_Trivial, !cir.ptr<!rec_Trivial>, ["__retval"]
  // CIR: %[[GET_A:.*]] = cir.get_member %[[RET_ALLOCA]][0] {name = "a"}
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0>
  // CIR: cir.store{{.*}} %[[ZERO]], %[[GET_A]]

  // LLVM-LABEL: ret_trivial
  // LLVM: %[[RET_ALLOCA:.*]] = alloca %union.Trivial
  // LLVM: store i32 0, ptr %[[RET_ALLOCA]]
