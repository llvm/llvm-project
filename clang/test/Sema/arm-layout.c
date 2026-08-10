// RUN: %clang_cc1 -triple armv7-unknown-unknown -target-abi apcs-gnu %s -verify
// RUN: %clang_cc1 -triple armv7-unknown-unknown -target-abi aapcs %s -verify
// expected-no-diagnostics

#define check(name, cond) int _##name##_check[(cond) ? 1 : -1]

struct s0 { char field0; double field1; };
#ifdef __ARM_EABI__
check(s0_size, sizeof(struct s0) == 16);
#else
check(s0_size, sizeof(struct s0) == 12);
#endif

struct s1 { char field0; long double field1; };
#ifdef __ARM_EABI__
check(s1_size, sizeof(struct s1) == 16);
#else
check(s1_size, sizeof(struct s1) == 12);
#endif

struct s2 {
  short field0;
  int field1 : 24;
  char field2;
};
#ifdef __ARM_EABI__
check(s2_size, sizeof(struct s2) == 8);
check(s2_offset_0, __builtin_offsetof(struct s2, field0) == 0);
check(s2_offset_1, __builtin_offsetof(struct s2, field2) == 7);
#else
check(s2_size, sizeof(struct s2) == 6);
check(s2_offset_0, __builtin_offsetof(struct s2, field0) == 0);
check(s2_offset_1, __builtin_offsetof(struct s2, field2) == 5);
#endif

struct s3 {
  short field0;
  int field1 : 24 __attribute__((aligned(4)));
  char field2;
};
check(s3_size, sizeof(struct s3) == 8);
check(s3_offset_0, __builtin_offsetof(struct s3, field0) == 0);
check(s3_offset_1, __builtin_offsetof(struct s3, field2) == 7);

struct s4 {
  int field0 : 4;
};
#ifdef __ARM_EABI__
check(s4_size, sizeof(struct s4) == 4);
check(s4_align, __alignof(struct s4) == 4);
#else
check(s4_size, sizeof(struct s4) == 1);
check(s4_align, __alignof(struct s4) == 1);
#endif
