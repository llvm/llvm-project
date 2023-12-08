// RUN: %clang_cc1 -S -emit-llvm -triple i386-linux -std=c++2a -Wno-unknown-pragmas %s -o - | FileCheck %s

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
