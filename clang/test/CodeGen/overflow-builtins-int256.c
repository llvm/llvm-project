// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Test overflow builtins with __int256_t and __uint256_t.

void overflowed(void);

// CHECK-LABEL: define {{.*}}@test_sadd_overflow_int256
// CHECK: call { i256, i1 } @llvm.sadd.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_sadd_overflow_int256(__int256_t x, __int256_t y) {
  __int256_t r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return (int)r;
}

// CHECK-LABEL: define {{.*}}@test_uadd_overflow_uint256
// CHECK: call { i256, i1 } @llvm.uadd.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_uadd_overflow_uint256(__uint256_t x, __uint256_t y) {
  __uint256_t r;
  if (__builtin_add_overflow(x, y, &r))
    overflowed();
  return (int)r;
}

// CHECK-LABEL: define {{.*}}@test_ssub_overflow_int256
// CHECK: call { i256, i1 } @llvm.ssub.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_ssub_overflow_int256(__int256_t x, __int256_t y) {
  __int256_t r;
  if (__builtin_sub_overflow(x, y, &r))
    overflowed();
  return (int)r;
}

// CHECK-LABEL: define {{.*}}@test_usub_overflow_uint256
// CHECK: call { i256, i1 } @llvm.usub.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_usub_overflow_uint256(__uint256_t x, __uint256_t y) {
  __uint256_t r;
  if (__builtin_sub_overflow(x, y, &r))
    overflowed();
  return (int)r;
}

// CHECK-LABEL: define {{.*}}@test_smul_overflow_int256
// CHECK: call { i256, i1 } @llvm.smul.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_smul_overflow_int256(__int256_t x, __int256_t y) {
  __int256_t r;
  if (__builtin_mul_overflow(x, y, &r))
    overflowed();
  return (int)r;
}

// CHECK-LABEL: define {{.*}}@test_umul_overflow_uint256
// CHECK: call { i256, i1 } @llvm.umul.with.overflow.i256(i256 %{{.+}}, i256 %{{.+}})
int test_umul_overflow_uint256(__uint256_t x, __uint256_t y) {
  __uint256_t r;
  if (__builtin_mul_overflow(x, y, &r))
    overflowed();
  return (int)r;
}
