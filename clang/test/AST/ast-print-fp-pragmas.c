// RUN: %clang_cc1 -triple x86_64-pc-linux -ast-print %s -o - | FileCheck %s

float func_1(float x, float y) {
#pragma STDC FENV_ACCESS ON
  if (x != 0) {
    return y;
  }
  return x + y;
}

// CHECK-LABEL: float func_1(float x, float y) {
// CHECK-NEXT:  #pragma STDC FENV_ACCESS ON
// CHECK-NEXT:      if (x != 0) {
// CHECK-NEXT:          return y;
// CHECK-NEXT:      }
// CHECK-NEXT:      return x + y;
// CHECK-NEXT:  }

float func_2(float x, float y) {
#pragma STDC FENV_ACCESS ON
  if (x != 0) {
  #pragma STDC FENV_ACCESS OFF
    return y;
  }
  return x + y;
}

// CHECK-LABEL: float func_2(float x, float y) {
// CHECK-NEXT:  #pragma STDC FENV_ACCESS ON
// CHECK-NEXT:      if (x != 0) {
// CHECK-NEXT:      #pragma STDC FENV_ACCESS OFF
// CHECK-NEXT:          return y;
// CHECK-NEXT:      }
// CHECK-NEXT:      return x + y;
// CHECK-NEXT:  }

float func_3(float x, float y) {
#pragma STDC FENV_ROUND FE_DOWNWARD
  return x + y;
}

// CHECK-LABEL: float func_3(float x, float y) {
// CHECK-NEXT:  #pragma STDC FENV_ROUND FE_DOWNWARD
// CHECK-NEXT:      return x + y;
// CHECK-NEXT:  }

float func_4(float x, float y, float z) {
#pragma STDC FENV_ACCESS ON
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ROUND FE_UPWARD
  if (z != 0) {
  #pragma STDC FENV_ACCESS OFF
  #pragma STDC FENV_ROUND FE_TOWARDZERO
    return z + x;
  }
  return x + y;
}

// CHECK-LABEL: float func_4(float x, float y, float z) {
// CHECK-NEXT:  #pragma STDC FENV_ACCESS ON
// CHECK-NEXT:  #pragma clang fp exceptions(maytrap)
// CHECK-NEXT:  #pragma STDC FENV_ROUND FE_UPWARD
// CHECK-NEXT:      if (z != 0) {
// CHECK-NEXT:      #pragma STDC FENV_ACCESS OFF
// CHECK-NEXT:      #pragma STDC FENV_ROUND FE_TOWARDZERO
// CHECK-NEXT:          return z + x;
// CHECK-NEXT:      }
// CHECK-NEXT:      return x + y;
// CHECK-NEXT:  }
