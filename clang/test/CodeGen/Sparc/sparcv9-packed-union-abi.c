// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - \
// RUN:   -fclang-abi-compat=23 | FileCheck %s --check-prefix=COMPAT

// GCC never passes or returns a union, or a struct with a packed field, in
// floating-point registers on SPARC64. A struct declared packed has all of its
// fields packed. Being packed carries over to the structs nested in a packed
// struct, but not to the structs a packed struct is nested in.
// See https://github.com/llvm/llvm-project/issues/39500 and
// https://github.com/llvm/llvm-project/issues/214840.

union u {
  float f;
  int i;
};

// CHECK-LABEL: define{{.*}} float @union_float(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} float @union_float(float inreg %x.coerce0, i32 inreg %x.coerce1)
float union_float(union u x) { return x.f; }

// CHECK-LABEL: define{{.*}} signext i32 @union_int(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} signext i32 @union_int(float inreg %x.coerce0, i32 inreg %x.coerce1)
int union_int(union u x) { return x.i; }

// CHECK-LABEL: define{{.*}} i64 @union_ret(float noundef %f)
// COMPAT-LABEL: define{{.*}} inreg { float, i32 } @union_ret(float noundef %f)
union u union_ret(float f) {
  union u x;
  x.f = f;
  return x;
}

union ud {
  double d;
  long l;
};

// CHECK-LABEL: define{{.*}} i64 @union_double(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} %union.ud @union_double(double %x.coerce)
union ud union_double(union ud x) { return x; }

// A union of structs is passed in integer registers.
union us {
  struct {
    float a, b;
  } s;
  int i;
};

// CHECK-LABEL: define{{.*}} i64 @union_struct(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @union_struct(float inreg %x.coerce0, float inreg %x.coerce1)
union us union_struct(union us x) { return x; }

// A union field is passed in integer registers, the other fields are not
// affected.
struct union_first {
  union u u;
  float g;
};

// CHECK-LABEL: define{{.*}} inreg { i32, float } @union_first(i32 inreg %x.coerce0, float inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @union_first(float inreg %x.coerce0, float inreg %x.coerce1)
struct union_first union_first(struct union_first x) { return x; }

struct union_second {
  float f;
  union u u;
};

// CHECK-LABEL: define{{.*}} inreg { float, i32 } @union_second(float inreg %x.coerce0, i32 inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @union_second(float inreg %x.coerce0, float inreg %x.coerce1)
struct union_second union_second(struct union_second x) { return x; }

struct packed_float_int {
  float f;
  int i;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} i64 @packed_float_int(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg %struct.packed_float_int @packed_float_int(float inreg %x.coerce0, i32 inreg %x.coerce1)
struct packed_float_int packed_float_int(struct packed_float_int x) {
  return x;
}

struct packed_double {
  double d;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} i64 @packed_double(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} %struct.packed_double @packed_double(double %x.coerce)
struct packed_double packed_double(struct packed_double x) { return x; }

struct packed_complex {
  _Complex float c;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} i64 @packed_complex(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @packed_complex(float inreg %x.coerce0, float inreg %x.coerce1)
struct packed_complex packed_complex(struct packed_complex x) { return x; }

// A struct nested in a packed struct is packed too.
struct floats {
  float f, g;
};

struct packed_outer {
  struct floats s;
  double d;
} __attribute__((packed));

// CHECK-LABEL: define{{.*}} { i64, i64 } @packed_outer(i64 %x.coerce0, i64 %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float, double } @packed_outer(float inreg %x.coerce0, float inreg %x.coerce1, double inreg %x.coerce2)
struct packed_outer packed_outer(struct packed_outer x) { return x; }

// A packed struct does not affect the struct it is nested in.
struct packed_float {
  float f;
} __attribute__((packed));

struct plain_outer {
  struct packed_float s;
  float g;
};

// CHECK-LABEL: define{{.*}} inreg { i32, float } @plain_outer(i32 inreg %x.coerce0, float inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float } @plain_outer(float inreg %x.coerce0, float inreg %x.coerce1)
struct plain_outer plain_outer(struct plain_outer x) { return x; }

// A single packed field makes the whole struct packed.
struct packed_field {
  float f;
  float g __attribute__((packed));
};

// CHECK-LABEL: define{{.*}} i64 @packed_field(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg %struct.packed_field @packed_field(float inreg %x.coerce0, float inreg %x.coerce1)
struct packed_field packed_field(struct packed_field x) { return x; }

struct packed_field_nested {
  struct floats s;
  int i __attribute__((packed));
};

// CHECK-LABEL: define{{.*}} { i64, i64 } @packed_field_nested(i64 %x.coerce0, i64 %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { float, float, i64 } @packed_field_nested(float inreg %x.coerce0, float inreg %x.coerce1, i64 inreg %x.coerce2)
struct packed_field_nested packed_field_nested(struct packed_field_nested x) {
  return x;
}

// The packed attribute has no effect on a byte-aligned field, so the struct is
// not packed.
struct packed_char {
  char c __attribute__((packed));
  float f;
};

// CHECK-LABEL: define{{.*}} inreg { i32, float } @packed_char(i32 inreg %x.coerce0, float inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg { i32, float } @packed_char(i32 inreg %x.coerce0, float inreg %x.coerce1)
struct packed_char packed_char(struct packed_char x) { return x; }

// But it does on a bit-field.
struct packed_char_bitfield {
  char c : 4 __attribute__((packed));
  float f;
};

// CHECK-LABEL: define{{.*}} i64 @packed_char_bitfield(i64 %x.coerce)
// COMPAT-LABEL: define{{.*}} inreg { i32, float } @packed_char_bitfield(i32 inreg %x.coerce0, float inreg %x.coerce1)
struct packed_char_bitfield packed_char_bitfield(struct packed_char_bitfield x) {
  return x;
}

// #pragma pack only reduces the alignment, it does not make the fields packed.
#pragma pack(1)
struct pragma_packed {
  float f;
  int i;
};
#pragma pack()

// CHECK-LABEL: define{{.*}} inreg %struct.pragma_packed @pragma_packed(float inreg %x.coerce0, i32 inreg %x.coerce1)
// COMPAT-LABEL: define{{.*}} inreg %struct.pragma_packed @pragma_packed(float inreg %x.coerce0, i32 inreg %x.coerce1)
struct pragma_packed pragma_packed(struct pragma_packed x) { return x; }

// A misaligned double is passed in integer registers, like any misaligned
// floating-point field. GCC passes it in the floating-point register of the
// float before it instead, overwriting the float, see
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=129107.
#pragma pack(1)
struct pragma_packed_double {
  float f;
  double d;
};
#pragma pack()

// CHECK-LABEL: define{{.*}} inreg { float, i32, i64 } @pragma_packed_double(float inreg %x.coerce0, i32 inreg %x.coerce1, i64 inreg %x.coerce2)
// COMPAT-LABEL: define{{.*}} inreg { float, i32, i64 } @pragma_packed_double(float inreg %x.coerce0, i32 inreg %x.coerce1, i64 inreg %x.coerce2)
struct pragma_packed_double pragma_packed_double(struct pragma_packed_double x) {
  return x;
}
