// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -x hip -fcuda-is-device -emit-llvm  %s \
// RUN:   -o - | FileCheck %s

constexpr static int OpCtrl()
{
    return 15 + 1;
}

constexpr static int RowMask()
{
    return 3 + 1;
}

constexpr static int BankMask()
{
    return 2 + 1;
}

constexpr static bool BountCtrl()
{
    return true & false;
}

// CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %1, i32 16, i32 0, i32 0, i1 false)
__attribute__((global)) void test_update_dpp_const_fold_imm_operand_2(int* out, int a, int b)
{
  *out = __builtin_amdgcn_update_dpp(a, b, OpCtrl(), 0, 0, false);
}

// CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %1, i32 0, i32 4, i32 0, i1 false)
__attribute__((global)) void test_update_dpp_const_fold_imm_operand_3(int* out, int a, int b)
{
  *out = __builtin_amdgcn_update_dpp(a, b, 0, RowMask(), 0, false);
}

// CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %1, i32 0, i32 0, i32 3, i1 false)
__attribute__((global)) void test_update_dpp_const_fold_imm_operand_4(int* out, int a, int b)
{
  *out = __builtin_amdgcn_update_dpp(a, b, 0, 0, BankMask(), false);
}

// CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %0, i32 %1, i32 0, i32 0, i32 0, i1 false)
__attribute__((global)) void test_update_dpp_const_fold_imm_operand_5(int* out, int a, int b)
{
  *out = __builtin_amdgcn_update_dpp(a, b, 0, 0, 0, BountCtrl());
}
