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

// OGCG: @__const.ret_outer.o = {{.*}}{ { i32, [4 x i8] }, i32, [4 x i8] } { { i32, [4 x i8] } zeroinitializer, i32 1, [4 x i8] zeroinitializer }

// In C mode, this does do zero padding.
vec3 ret_vec3() {
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

union needs_padding {
  int a;
  long long b;
};
struct outer {
  union needs_padding np;
  int x;
};

struct outer ret_outer() {
  struct outer o = {{}, 1};
  return o;

  // CIR-LABEL: ret_outer
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca !rec_outer, !cir.ptr<!rec_outer>, ["__retval"]
  // CIR: %[[BITCAST:.*]] = cir.cast bitcast %0 : !cir.ptr<!rec_outer> -> !cir.ptr<!{{.*}}>
  // CIR: %[[RECORD:.*]] = cir.const #cir.const_record<{#cir.zero : !{{.*}}, #cir.int<1> : !s32i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>}> 
  // CIR: cir.store {{.*}}%[[RECORD]], %[[BITCAST]] 

  // LLVM-LABEL: ret_outer
  // LLVM: %[[RET_ALLOCA:.*]] = alloca %struct.outer
  // LLVMCIR: store { { i32, [4 x i8] }, i32, [4 x i8] } { { i32, [4 x i8] } zeroinitializer, i32 1, [4 x i8] zeroinitializer }, ptr %[[RET_ALLOCA]]
  // OGCG: call void @llvm.memcpy{{.*}}(ptr{{.*}}%[[RET_ALLOCA]], ptr {{.*}}@__const.ret_outer.o, i64 16, i1 false)
}
