// RUN: %clang_cc1 -emit-llvm -triple i386-linux -std=c++2a -Wno-unknown-pragmas %s -o - | FileCheck %s

// nextUp(1.F) == 0x1.000002p0F

constexpr float add_round_down(float x, float y) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  float res = x;
  res += y;
  return res;
}

constexpr float add_round_up(float x, float y) {
  #pragma STDC FENV_ROUND FE_UPWARD
  float res = x;
  res += y;
  return res;
}

float V1 = add_round_down(1.0F, 0x0.000001p0F);
float V2 = add_round_up(1.0F, 0x0.000001p0F);
// CHECK: @V1 = {{.*}} float 1.000000e+00
// CHECK: @V2 = {{.*}} float 0x3FF0000020000000

constexpr float add_cast_round_down(float x, double y) {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  float res = x;
  res += y;
  return res;
}

constexpr float add_cast_round_up(float x, double y) {
  #pragma STDC FENV_ROUND FE_UPWARD
  float res = x;
  res += y;
  return res;
}

float V3 = add_cast_round_down(1.0F, 0x0.000001p0F);
float V4 = add_cast_round_up(1.0F, 0x0.000001p0F);

// CHECK: @V3 = {{.*}} float 1.000000e+00
// CHECK: @V4 = {{.*}} float 0x3FF0000020000000

// The next three variables use the same function as initializer, only rounding
// modes differ.

float V5 = []() -> float {
  return [](float x, float y)->float {
    #pragma STDC FENV_ROUND FE_UPWARD
    return x + y;
  }([](float x, float y) -> float {
      #pragma STDC FENV_ROUND FE_UPWARD
      return x + y;
    }(1.0F, 0x0.000001p0F),
  0x0.000001p0F);
}();
// CHECK: @V5 = {{.*}} float 0x3FF0000040000000

float V6 = []() -> float {
  return [](float x, float y)->float {
    #pragma STDC FENV_ROUND FE_DOWNWARD
    return x + y;
  }([](float x, float y) -> float {
      #pragma STDC FENV_ROUND FE_UPWARD
      return x + y;
    }(1.0F, 0x0.000001p0F),
  0x0.000001p0F);
}();
// CHECK: @V6 = {{.*}} float 0x3FF0000020000000

float V7 = []() -> float {
  return [](float x, float y)->float {
    #pragma STDC FENV_ROUND FE_DOWNWARD
    return x + y;
  }([](float x, float y) -> float {
      #pragma STDC FENV_ROUND FE_DOWNWARD
      return x + y;
    }(1.0F, 0x0.000001p0F),
  0x0.000001p0F);
}();
// CHECK: @V7 = {{.*}} float 1.000000e+00

#pragma STDC FENV_ROUND FE_DYNAMIC

template<float V> struct L {
  constexpr L() : value(V) {}
  float value;
};

#pragma STDC FENV_ROUND FE_DOWNWARD
L<0.1F> val_d;
// CHECK: @val_d = {{.*}} { float 0x3FB9999980000000 }

#pragma STDC FENV_ROUND FE_UPWARD
L<0.1F> val_u;
// CHECK: @val_u = {{.*}} { float 0x3FB99999A0000000 }


// Check literals in macros.

#pragma STDC FENV_ROUND FE_DOWNWARD
#define CONSTANT_0_1 0.1F

#pragma STDC FENV_ROUND FE_UPWARD
float C1_ru = CONSTANT_0_1;
// CHECK: @C1_ru = {{.*}} float 0x3FB99999A0000000

#pragma STDC FENV_ROUND FE_DOWNWARD
float C1_rd = CONSTANT_0_1;
// CHECK: @C1_rd = {{.*}} float 0x3FB9999980000000

#pragma STDC FENV_ROUND FE_DOWNWARD
#define PRAGMA(x) _Pragma(#x)
#define CONSTANT_0_1_RM(v, rm) ([](){ PRAGMA(STDC FENV_ROUND rm); return v; }())

#pragma STDC FENV_ROUND FE_UPWARD
float C2_rd = CONSTANT_0_1_RM(0.1F, FE_DOWNWARD);
float C2_ru = CONSTANT_0_1_RM(0.1F, FE_UPWARD);
// CHECK: @C2_rd = {{.*}} float 0x3FB9999980000000
// CHECK: @C2_ru = {{.*}} float 0x3FB99999A0000000

#pragma STDC FENV_ROUND FE_DOWNWARD
float C3_rd = CONSTANT_0_1_RM(0.1F, FE_DOWNWARD);
float C3_ru = CONSTANT_0_1_RM(0.1F, FE_UPWARD);
// CHECK: @C3_rd = {{.*}} float 0x3FB9999980000000
// CHECK: @C3_ru = {{.*}} float 0x3FB99999A0000000

// Check literals in template instantiations.

#pragma STDC FENV_ROUND FE_DYNAMIC

template<typename T, T C>
constexpr T foo() {
  return C;
}

#pragma STDC FENV_ROUND FE_DOWNWARD
float var_d = foo<float, 0.1F>();
// CHECK: @var_d = {{.*}} float 0x3FB9999980000000

#pragma STDC FENV_ROUND FE_UPWARD
float var_u = foo<float, 0.1F>();
// CHECK: @var_u = {{.*}} float 0x3FB99999A0000000

#pragma STDC FENV_ROUND FE_DYNAMIC

template<typename T, T f> void foo2() {
  T Val = f;
}

void func_01() {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  foo2<float, 0.1f>();
}

void func_02() {
  #pragma STDC FENV_ROUND FE_UPWARD
  foo2<float, 0.1f>();
}

// CHECK-LABEL: define {{.*}} void @_Z4foo2IfTnT_Lf3dccccccEEvv()
// CHECK:         store float 0x3FB9999980000000, ptr

// CHECK-LABEL: define {{.*}} void @_Z4foo2IfTnT_Lf3dcccccdEEvv()
// CHECK:         store float 0x3FB99999A0000000, ptr


#pragma STDC FENV_ROUND FE_DOWNWARD
template <int C>
float tfunc_01() {
  return 0.1F;  // Must be 0x3FB9999980000000 in all instantiations.
}
template float tfunc_01<0>();
// CHECK-LABEL: define {{.*}} float @_Z8tfunc_01ILi0EEfv()
// CHECK:         ret float 0x3FB9999980000000

#pragma STDC FENV_ROUND FE_UPWARD
template float tfunc_01<1>();
// CHECK-LABEL: define {{.*}} float @_Z8tfunc_01ILi1EEfv()
// CHECK:         ret float 0x3FB9999980000000

template<> float tfunc_01<2>() {
  return 0.1F;
}
// CHECK-LABEL: define {{.*}} float @_Z8tfunc_01ILi2EEfv()
// CHECK:         ret float 0x3FB99999A0000000
