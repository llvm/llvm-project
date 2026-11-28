// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - \
// RUN:   -fclang-abi-compat=23 | FileCheck %s --check-prefix=COMPAT

// GCC treats base classes like fields when deciding whether a struct with a
// packed field, or a union, is passed in integer registers on SPARC64.

struct floats {
  float f, g;
};

struct packed_base {
  float f;
} __attribute__((packed));

// Only the packed base is passed in integer registers.
struct derived_from_packed : packed_base {
  float g;
};

// CHECK-LABEL: define{{.*}} inreg { i32, float } @_Z19derived_from_packed19derived_from_packed(i32 inreg %x.coerce0, float inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @_Z19derived_from_packed19derived_from_packed(float inreg %x.coerce0, float inreg %x.coerce1)
derived_from_packed derived_from_packed(derived_from_packed x) { return x; }

struct base_float {
  float f;
};

// A packed class makes its base classes packed.
struct packed_derived : base_float {
  float g;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} i64 @_Z14packed_derived14packed_derived(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @_Z14packed_derived14packed_derived(float inreg %x.coerce0, float inreg %x.coerce1)
packed_derived packed_derived(packed_derived x) { return x; }

struct base_with_packed_field {
  float f;
  float g __attribute__((packed));
};

struct derived_from_packed_field : base_with_packed_field {
  double d;
};

// CHECK-LABEL: define{{.*}} { i64, double } @_Z25derived_from_packed_field25derived_from_packed_field(i64 %x.coerce0, double %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float, double } @_Z25derived_from_packed_field25derived_from_packed_field(float inreg %x.coerce0, float inreg %x.coerce1, double inreg %x.coerce2)
derived_from_packed_field derived_from_packed_field(derived_from_packed_field x) {
  return x;
}

union u {
  float f;
  int i;
};

struct base_with_union {
  u x;
};

struct derived_from_union_holder : base_with_union {
  float g;
};

// CHECK-LABEL: define{{.*}} inreg { i32, float } @_Z25derived_from_union_holder25derived_from_union_holder(i32 inreg %x.coerce0, float inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @_Z25derived_from_union_holder25derived_from_union_holder(float inreg %x.coerce0, float inreg %x.coerce1)
derived_from_union_holder derived_from_union_holder(derived_from_union_holder x) {
  return x;
}

// A packed class does not pack a field of a non-POD class type, so this class
// is not packed.
struct non_pod {
  non_pod() {}
  float f;
};

struct packed_with_non_pod {
  non_pod n;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} inreg { float, i32 } @_Z19packed_with_non_pod19packed_with_non_pod(float inreg %x.coerce0, i32 inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, i32 } @_Z19packed_with_non_pod19packed_with_non_pod(float inreg %x.coerce0, i32 inreg %x.coerce1)
packed_with_non_pod packed_with_non_pod(packed_with_non_pod x) { return x; }

// A defaulted constructor does not make a class non-POD, so this class is
// packed.
struct pod_defaulted {
  pod_defaulted() = default;
  float f;
};

struct packed_with_pod_defaulted {
  pod_defaulted n;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} i64 @_Z25packed_with_pod_defaulted25packed_with_pod_defaulted(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg { float, i32 } @_Z25packed_with_pod_defaulted25packed_with_pod_defaulted(float inreg %x.coerce0, i32 inreg %x.coerce1)
packed_with_pod_defaulted
packed_with_pod_defaulted(packed_with_pod_defaulted x) {
  return x;
}

// Plain base classes are not affected.
struct derived_from_floats : floats {
  float h;
};

// CHECK-LABEL: define{{.*}} inreg { float, float, float, i32 } @_Z19derived_from_floats19derived_from_floats(float inreg %x.coerce0, float inreg %x.coerce1, float inreg %x.coerce2, i32 inreg %x.coerce3)
// COMPAT-LABEL: define{{.*}} inreg { float, float, float, i32 } @_Z19derived_from_floats19derived_from_floats(float inreg %x.coerce0, float inreg %x.coerce1, float inreg %x.coerce2, i32 inreg %x.coerce3)
derived_from_floats derived_from_floats(derived_from_floats x) { return x; }
