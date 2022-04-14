// RUN: %clang_cc1 -S -triple x86_64-linux-gnu -emit-llvm %s -o - | \
// RUN:   FileCheck %s --implicit-check-not "call void @llvm.set.rounding" --implicit-check-not "call i32 @llvm.get.rounding"

float func_rz_ru(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_UPWARD
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_rz_ru
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 2)
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)


float func_rz_rz(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_TOWARDZERO
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_rz_rz
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)

float func_rne_rne(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TONEAREST
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_TONEAREST
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_rne_rne
// CHECK:  fmul
// CHECK:  fadd
// CHECK:  fsub

float func_rz_dyn_noacc(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_DYNAMIC
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_rz_dyn_noacc
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)


float func_rz_dyn(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_TOWARDZERO
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_DYNAMIC
    #pragma STDC FENV_ACCESS ON
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_rz_dyn
// CHECK:  call void @llvm.set.rounding(i32 0)
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.towardzero", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)


float func_dyn_ru(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  #pragma STDC FENV_ACCESS ON
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_UPWARD
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_dyn_ru
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK:  [[RM:%[0-9]+]] = call i32 @llvm.get.rounding()
// CHECK:  call void @llvm.set.rounding(i32 2)
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// CHECK:  call void @llvm.set.rounding(i32 [[RM]])
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")


float func_dyn_dyn(float w, float x, float y, float z) {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  #pragma STDC FENV_ACCESS ON
  #pragma clang fp exceptions(ignore)
  float result = x * y;
  {
    #pragma STDC FENV_ROUND FE_DYNAMIC
    result += z;
  }
  return result - w;
}

// CHECK-LABEL: @func_dyn_dyn
// CHECK:  call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")


#pragma STDC FENV_ROUND FE_DOWNWARD
float func_glb_rd(float x, float y) {
  return x + y;
}

// CHECK-LABEL: func_glb_rd
// CHECK:  call void @llvm.set.rounding(i32 3)
// CHECK:  call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)

float func_glb_rd2(float x, float y) {
  return x - y;
}

// CHECK-LABEL: func_glb_rd2
// CHECK:  call void @llvm.set.rounding(i32 3)
// CHECK:  call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)

#pragma STDC FENV_ROUND FE_DOWNWARD
float func_cvt_rd(int x) {
  return x;
}

// CHECK-LABEL: @func_cvt_rd
// CHECK:  call void @llvm.set.rounding(i32 3)
// CHECK:  call float @llvm.experimental.constrained.sitofp.f32.i32({{.*}}, metadata !"round.downward", metadata !"fpexcept.ignore")
// CHECK:  call void @llvm.set.rounding(i32 1)
