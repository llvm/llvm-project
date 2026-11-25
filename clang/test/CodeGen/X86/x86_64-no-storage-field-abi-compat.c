// RUN: %clang_cc1 -w -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefixes=CHECK,NEW
// RUN: %clang_cc1 -w -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm -fclang-abi-compat=23 %s -o - | \
// RUN:   FileCheck %s -check-prefixes=CHECK,COMPAT
// RUN: %clang_cc1 -w -triple x86_64-scei-ps4 -target-feature +avx -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefixes=CHECK,COMPAT
// RUN: %clang_cc1 -w -triple x86_64-sie-ps5 -target-feature +avx -emit-llvm %s -o - | \
// RUN:   FileCheck %s -check-prefixes=CHECK,COMPAT

// A field that occupies no storage -- a flexible array member or an array of
// length zero -- is NO_CLASS regardless of its offset, like a zero-width
// bit-field. Under -fclang-abi-compat=23 (and on PlayStation) a record with a
// flexible array member is passed in memory, and a zero-length array is
// classified like any other field.

// A zero-length array at a non-zero offset in an eightbyte was already
// NO_CLASS.
struct s4 { float x; int a[0]; };

// CHECK-LABEL: define{{.*}} float @get_s4()
struct s4 get_s4(void) { return (struct s4){0}; }
// CHECK-LABEL: define{{.*}} void @put_s4(float %a.coerce)
void put_s4(struct s4 a) {}

// Flexible array member.
struct s5 { float x; int a[]; };

// NEW-LABEL:    define{{.*}} float @get_s5()
// COMPAT-LABEL: define{{.*}} void @get_s5(ptr {{.*}}sret(%struct.s5)
struct s5 get_s5(void) { return (struct s5){0}; }
// NEW-LABEL:    define{{.*}} void @put_s5(float %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_s5(ptr noundef byval(%struct.s5)
void put_s5(struct s5 a) {}

// Flexible array member that raises the alignment, and so the size, of the
// record. The tail padding is NO_CLASS too.
struct s6 { char c; double a[]; };

// NEW-LABEL:    define{{.*}} i8 @get_s6()
// COMPAT-LABEL: define{{.*}} void @get_s6(ptr {{.*}}sret(%struct.s6)
struct s6 get_s6(void) { return (struct s6){0}; }
// NEW-LABEL:    define{{.*}} void @put_s6(i8 %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_s6(ptr noundef byval(%struct.s6)
void put_s6(struct s6 a) {}

// Union with a flexible array member (GNU extension).
union u7 { float f; int a[]; };

// NEW-LABEL:    define{{.*}} float @get_u7()
// COMPAT-LABEL: define{{.*}} void @get_u7(ptr {{.*}}sret(%union.u7)
union u7 get_u7(void) { return (union u7){0}; }
// NEW-LABEL:    define{{.*}} void @put_u7(float %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_u7(ptr noundef byval(%union.u7)
void put_u7(union u7 a) {}

// Struct whose last member is a struct with a flexible array member (GNU
// extension).
struct s8 { double d; struct s5 s; };

// NEW-LABEL:    define{{.*}} { double, float } @get_s8()
// COMPAT-LABEL: define{{.*}} void @get_s8(ptr {{.*}}sret(%struct.s8)
struct s8 get_s8(void) { return (struct s8){0}; }
// NEW-LABEL:    define{{.*}} void @put_s8(double %a.coerce0, float %a.coerce1)
// COMPAT-LABEL: define{{.*}} void @put_s8(ptr noundef byval(%struct.s8)
void put_s8(struct s8 a) {}

// A zero-length array at an offset that is not aligned for its element type
// no longer makes the record unaligned.
struct __attribute__((packed)) s9 { char c; int a[0]; };

// NEW-LABEL:    define{{.*}} i8 @get_s9()
// COMPAT-LABEL: define{{.*}} void @get_s9(ptr {{.*}}sret(%struct.s9)
struct s9 get_s9(void) { return (struct s9){0}; }
// NEW-LABEL:    define{{.*}} void @put_s9(i8 %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_s9(ptr noundef byval(%struct.s9)
void put_s9(struct s9 a) {}

// A zero-length array no longer stops a record holding a single 256-bit
// vector from being passed in a YMM register.
typedef float v8f __attribute__((vector_size(32)));
struct s10 { v8f v; float a[0]; };

// NEW-LABEL:    define{{.*}} <8 x float> @get_s10()
// COMPAT-LABEL: define{{.*}} void @get_s10(ptr {{.*}}sret(%struct.s10)
struct s10 get_s10(void) { return (struct s10){0}; }
// NEW-LABEL:    define{{.*}} void @put_s10(<8 x float> %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_s10(ptr noundef byval(%struct.s10)
void put_s10(struct s10 a) {}

// Same for a flexible array member. (The vector is coerced to a generic 256-bit
// vector type, which is passed in the same YMM register.)
struct s11 { v8f v; float a[]; };

// NEW-LABEL:    define{{.*}} <4 x double> @get_s11()
// COMPAT-LABEL: define{{.*}} void @get_s11(ptr {{.*}}sret(%struct.s11)
struct s11 get_s11(void) { return (struct s11){0}; }
// NEW-LABEL:    define{{.*}} void @put_s11(<4 x double> %a.coerce)
// COMPAT-LABEL: define{{.*}} void @put_s11(ptr noundef byval(%struct.s11)
void put_s11(struct s11 a) {}
