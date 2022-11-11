// RUN: %clang_cc1 -triple loongarch64 -target-feature +f -target-feature +d -target-abi lp64d \
// RUN:   -emit-llvm %s -o - | FileCheck %s

#include <stdint.h>

/// Ensure that fields inherited from a parent struct are treated in the same
/// way as fields directly in the child for the purposes of LoongArch ABI rules.

struct parent1_int32_s {
  int32_t i1;
};

struct child1_int32_s : parent1_int32_s {
  int32_t i2;
};

// CHECK-LABEL: define{{.*}} i64 @_Z30int32_int32_struct_inheritance14child1_int32_s(i64 %a.coerce)
struct child1_int32_s int32_int32_struct_inheritance(struct child1_int32_s a) {
  return a;
}

struct parent2_int32_s {
  int32_t i1;
};

struct child2_float_s : parent2_int32_s {
  float f1;
};

// CHECK-LABEL: define{{.*}} { i32, float } @_Z30int32_float_struct_inheritance14child2_float_s(i32 %0, float %1)
struct child2_float_s int32_float_struct_inheritance(struct child2_float_s a) {
  return a;
}

struct parent3_float_s {
  float f1;
};

struct child3_int64_s : parent3_float_s {
  int64_t i1;
};

// CHECK-LABEL: define{{.*}} { float, i64 } @_Z30float_int64_struct_inheritance14child3_int64_s(float %0, i64 %1)
struct child3_int64_s float_int64_struct_inheritance(struct child3_int64_s a) {
  return a;
}

struct parent4_double_s {
  double d1;
};

struct child4_double_s : parent4_double_s {
  double d1;
};

// CHECK-LABEL: define{{.*}} { double, double } @_Z32double_double_struct_inheritance15child4_double_s(double %0, double %1)
struct child4_double_s double_double_struct_inheritance(struct child4_double_s a) {
  return a;
}

/// When virtual inheritance is used, the resulting struct isn't eligible for
/// passing in registers.

struct parent5_virtual_s {
  int32_t i1;
};

struct child5_virtual_s : virtual parent5_virtual_s {
  float f1;
};

// CHECK-LABEL: define{{.*}} void @_ZN16child5_virtual_sC1EOS_(ptr noundef nonnull align 8 dereferenceable(12) %this, ptr noundef nonnull align 8 dereferenceable(12) %0)
struct child5_virtual_s int32_float_virtual_struct_inheritance(struct child5_virtual_s a) {
  return a;
}

/// Check for correct lowering in the presence of diamond inheritance.

struct parent6_float_s {
  float f1;
};

struct child6a_s : parent6_float_s {
};

struct child6b_s : parent6_float_s {
};

struct grandchild_6_s : child6a_s, child6b_s {
};

// CHECK-LABEL: define{{.*}} { float, float } @_Z38float_float_diamond_struct_inheritance14grandchild_6_s(float %0, float %1)
struct grandchild_6_s float_float_diamond_struct_inheritance(struct grandchild_6_s a) {
  return a;
}
