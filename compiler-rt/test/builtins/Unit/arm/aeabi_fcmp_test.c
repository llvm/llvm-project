// REQUIRES: arm-target-arch || armv4t-target-arch
// RUN: %clang_builtins -mthumb %s %librt -o %t && %run %t

#include <math.h>

extern int __aeabi_fcmpeq(float, float);
extern int __aeabi_fcmplt(float, float);
extern int __aeabi_fcmple(float, float);
extern int __aeabi_fcmpge(float, float);
extern int __aeabi_fcmpgt(float, float);
extern int __aeabi_dcmpeq(double, double);
extern int __aeabi_dcmplt(double, double);
extern int __aeabi_dcmple(double, double);
extern int __aeabi_dcmpge(double, double);
extern int __aeabi_dcmpgt(double, double);

#define CHECK_COMPARE(name, true_a, true_b, false_a, false_b)                  \
  do {                                                                         \
    if (!name(true_a, true_b))                                                 \
      return __LINE__;                                                         \
    if (name(false_a, false_b))                                                \
      return __LINE__;                                                         \
    if (name(NAN, false_b))                                                    \
      return __LINE__;                                                         \
  } while (0)

int main(void) {
  CHECK_COMPARE(__aeabi_fcmpeq, 1.0f, 1.0f, 1.0f, 2.0f);
  CHECK_COMPARE(__aeabi_fcmplt, 1.0f, 2.0f, 2.0f, 1.0f);
  CHECK_COMPARE(__aeabi_fcmple, 1.0f, 1.0f, 2.0f, 1.0f);
  CHECK_COMPARE(__aeabi_fcmpge, 2.0f, 1.0f, 1.0f, 2.0f);
  CHECK_COMPARE(__aeabi_fcmpgt, 2.0f, 1.0f, 1.0f, 1.0f);

  CHECK_COMPARE(__aeabi_dcmpeq, 1.0, 1.0, 1.0, 2.0);
  CHECK_COMPARE(__aeabi_dcmplt, 1.0, 2.0, 2.0, 1.0);
  CHECK_COMPARE(__aeabi_dcmple, 1.0, 1.0, 2.0, 1.0);
  CHECK_COMPARE(__aeabi_dcmpge, 2.0, 1.0, 1.0, 2.0);
  CHECK_COMPARE(__aeabi_dcmpgt, 2.0, 1.0, 1.0, 1.0);
  return 0;
}
