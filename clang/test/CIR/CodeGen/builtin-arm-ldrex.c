// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

struct twoFldT {
  char a, b;
};
// CIR: !ty_twoFldT = !cir.struct<struct "twoFldT" {!cir.int<s, 8>, !cir.int<s, 8>}
int test_ldrex(char *addr, long long *addr64, float *addrfloat) {
// CIR-LABEL: @test_ldrex
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
// CIR: [[INTRES0:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" {{%[0-9]+}} : (!cir.ptr<!s8i>) -> !s64i 
// CIR: [[CAST0:%.*]] = cir.cast(integral, [[INTRES0]] : !s64i), !s8i 
// CIR: [[CAST_I32:%.*]] = cir.cast(integral, [[CAST0]] : !s8i), !s32i

  sum += __builtin_arm_ldrex((short *)addr);
// CIR: [[INTRES1:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" {{%[0-9]+}} : (!cir.ptr<!s16i>) -> !s64i
// CIR: [[CAST1:%.*]] = cir.cast(integral, [[INTRES1]] : !s64i), !s16i 
// CIR: [[CAST_I16:%.*]] = cir.cast(integral, [[CAST1]] : !s16i), !s32i

  sum += __builtin_arm_ldrex((int *)addr);
// CIR: [[INTRES2:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" {{%[0-9]+}} : (!cir.ptr<!s32i>) -> !s64i
// CIR: [[CAST2:%.*]] = cir.cast(integral, [[INTRES2]] : !s64i), !s32i

  sum += __builtin_arm_ldrex((long long *)addr);
// CIR: [[INTRES3:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" {{%[0-9]+}} : (!cir.ptr<!s64i>) -> !s64i

  sum += __builtin_arm_ldrex(addr64);
// CIR: [[INTRES4:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" {{%[0-9]+}} : (!cir.ptr<!s64i>) -> !s64i


  sum += *__builtin_arm_ldrex((int **)addr);
// CIR: [[INTRES5:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr"  {{%[0-9]+}} : (!cir.ptr<!cir.ptr<!s32i>>) -> !s64i

  sum += __builtin_arm_ldrex((struct twoFldT **)addr)->a;
// CIR: [[INTRES6:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr"  {{%[0-9]+}} : (!cir.ptr<!cir.ptr<!ty_twoFldT>>) -> !s64i
// CIR: [[CAST3:%.*]] = cir.cast(int_to_ptr, [[INTRES6]] : !s64i), !cir.ptr<!ty_twoFldT>
// CIR: [[MEMBER_A:%.*]] = cir.get_member [[CAST3]][0] {name = "a"} : !cir.ptr<!ty_twoFldT> -> !cir.ptr<!s8i>


 // TODO: Uncomment next 2 lines, add tests when floating result type supported
 // sum += __builtin_arm_ldrex(addrfloat);

 // sum += __builtin_arm_ldrex((double *)addr);


  return sum;
}
