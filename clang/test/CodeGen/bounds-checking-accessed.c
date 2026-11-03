// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s

// -fsanitize=array-bounds allows an index one past the end of an array only
// when just the address is formed ('&a[i]', checked with 'ule'). When the
// element is read or written, the index must name an element ('ult'). That
// includes the operand of '++' and '--', '_Complex' and '_Atomic' elements,
// components of an element ('__real__'), and parenthesized subscripts.

int a[4];
_Atomic int at[4];
_Complex double c[4];

// CHECK-LABEL: define {{.*}} @test_post_inc(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_post_inc(int i) { a[i]++; }

// CHECK-LABEL: define {{.*}} @test_post_dec(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_post_dec(int i) { a[i]--; }

// CHECK-LABEL: define {{.*}} @test_pre_inc(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_pre_inc(int i) { ++a[i]; }

// CHECK-LABEL: define {{.*}} @test_pre_dec(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_pre_dec(int i) { --a[i]; }

// CHECK-LABEL: define {{.*}} @test_paren_inc(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_paren_inc(int i) { (a[i])++; }

// CHECK-LABEL: define {{.*}} @test_paren_assign(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_paren_assign(int i) { (a[i]) = 1; }

// CHECK-LABEL: define {{.*}} @test_atomic_inc(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_atomic_inc(int i) { at[i]++; }

// CHECK-LABEL: define {{.*}} @test_complex_inc(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_complex_inc(int i) { c[i]++; }

// CHECK-LABEL: define {{.*}} @test_complex_compound_assign(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_complex_compound_assign(int i) { c[i] += 1.0; }

// CHECK-LABEL: define {{.*}} @test_complex_assign(
// CHECK:       icmp ult i64 {{%.*}}, 4
void test_complex_assign(int i) { c[i] = 1.0; }

// CHECK-LABEL: define {{.*}} @test_complex_load(
// CHECK:       icmp ult i64 {{%.*}}, 4
_Complex double test_complex_load(int i) { return c[i]; }

// CHECK-LABEL: define {{.*}} @test_complex_real(
// CHECK:       icmp ult i64 {{%.*}}, 4
double test_complex_real(int i) { return __real__ c[i]; }

// A component of an element, like a member of one, only exists if the element
// does.
// CHECK-LABEL: define {{.*}} @test_complex_real_address(
// CHECK:       icmp ult i64 {{%.*}}, 4
double *test_complex_real_address(int i) { return &__real__ c[i]; }

// Forming the address of the element one past the end is allowed.
// CHECK-LABEL: define {{.*}} @test_address(
// CHECK:       icmp ule i64 {{%.*}}, 4
int *test_address(int i) { return &a[i]; }

// CHECK-LABEL: define {{.*}} @test_paren_address(
// CHECK:       icmp ule i64 {{%.*}}, 4
int *test_paren_address(int i) { return &(a[i]); }

// The same applies to 'counted_by' flexible array members and pointers.
struct fam {
  int count;
  int array[] __attribute__((counted_by(count)));
};

struct ptr {
  int count;
  int *buf __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_counted_by_fam_inc(
// CHECK:       icmp ult i32 {{%.*}}, %.counted_by.load
void test_counted_by_fam_inc(struct fam *p, int i) { p->array[i]++; }

// CHECK-LABEL: define {{.*}} @test_counted_by_ptr_inc(
// CHECK:       icmp ult i32 {{%.*}}, %.counted_by.load
void test_counted_by_ptr_inc(struct ptr *p, int i) { --p->buf[i]; }
