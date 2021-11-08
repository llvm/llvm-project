// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - | FileCheck %s
#include <stdarg.h>

// CHECK-LABEL: define{{.*}} void @f_void()
void f_void(void) {}

// Arguments and return values smaller than the word size are extended.

// CHECK-LABEL: define{{.*}} signext i32 @f_int_1(i32 signext %x)
int f_int_1(int x) { return x; }

// CHECK-LABEL: define{{.*}} zeroext i32 @f_int_2(i32 zeroext %x)
unsigned f_int_2(unsigned x) { return x; }

// CHECK-LABEL: define{{.*}} i64 @f_int_3(i64 %x)
long long f_int_3(long long x) { return x; }

// CHECK-LABEL: define{{.*}} signext i8 @f_int_4(i8 signext %x)
char f_int_4(char x) { return x; }

// CHECK-LABEL: define{{.*}} fp128 @f_ld(fp128 %x)
long double f_ld(long double x) { return x; }

// Small structs are passed in registers.
struct small {
  int *a, *b;
};

// CHECK-LABEL: define{{.*}} %struct.small @f_small(i32* %x.coerce0, i32* %x.coerce1)
struct small f_small(struct small x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

// Medium-sized structs are passed indirectly, but can be returned in registers.
struct medium {
  int *a, *b;
  int *c, *d;
};

// CHECK-LABEL: define{{.*}} %struct.medium @f_medium(%struct.medium* %x)
struct medium f_medium(struct medium x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

// Large structs are also returned indirectly.
struct large {
  int *a, *b;
  int *c, *d;
  int x;
};

// CHECK-LABEL: define{{.*}} void @f_large(%struct.large* noalias sret(%struct.large) align 8 %agg.result, %struct.large* %x)
struct large f_large(struct large x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

// A 64-bit struct fits in a register.
struct reg {
  int a, b;
};

// CHECK-LABEL: define{{.*}} i64 @f_reg(i64 %x.coerce)
struct reg f_reg(struct reg x) {
  x.a += x.b;
  return x;
}

// Structs with mixed int and float parts require the inreg attribute.
struct mixed {
  int a;
  float b;
};

// CHECK-LABEL: define{{.*}} inreg %struct.mixed @f_mixed(i32 inreg %x.coerce0, float inreg %x.coerce1)
struct mixed f_mixed(struct mixed x) {
  x.a += 1;
  return x;
}

// Struct with padding.
struct mixed2 {
  int a;
  double b;
};

// CHECK: define{{.*}} { i64, double } @f_mixed2(i64 %x.coerce0, double %x.coerce1)
// CHECK: store i64 %x.coerce0
// CHECK: store double %x.coerce1
struct mixed2 f_mixed2(struct mixed2 x) {
  x.a += 1;
  return x;
}

// Struct with single element and padding in passed in the high bits of a
// register.
struct tiny {
  char a;
};

// CHECK-LABEL: define{{.*}} i64 @f_tiny(i64 %x.coerce)
// CHECK: %[[HB:[^ ]+]] = lshr i64 %x.coerce, 56
// CHECK: = trunc i64 %[[HB]] to i8
struct tiny f_tiny(struct tiny x) {
  x.a += 1;
  return x;
}

// CHECK-LABEL: define{{.*}} void @call_tiny()
// CHECK: %[[XV:[^ ]+]] = zext i8 %{{[^ ]+}} to i64
// CHECK: %[[HB:[^ ]+]] = shl i64 %[[XV]], 56
// CHECK: = call i64 @f_tiny(i64 %[[HB]])
void call_tiny() {
  struct tiny x = { 1 };
  f_tiny(x);
}

// CHECK-LABEL: define{{.*}} signext i32 @f_variable(i8* %f, ...)
// CHECK: %ap = alloca i8*
// CHECK: call void @llvm.va_start
//
int f_variable(char *f, ...) {
  int s = 0;
  char c;
  va_list ap;
  va_start(ap, f);
  while ((c = *f++)) switch (c) {

// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK-DAG: %[[NXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 8
// CHECK-DAG: store i8* %[[NXT]], i8** %ap
// CHECK-DAG: %[[EXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 4
// CHECK-DAG: %[[ADR:[^ ]+]] = bitcast i8* %[[EXT]] to i32*
// CHECK-DAG: load i32, i32* %[[ADR]]
// CHECK: br
  case 'i':
    s += va_arg(ap, int);
    break;

// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK-DAG: %[[NXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 8
// CHECK-DAG: store i8* %[[NXT]], i8** %ap
// CHECK-DAG: %[[ADR:[^ ]+]] = bitcast i8* %[[CUR]] to i64*
// CHECK-DAG: load i64, i64* %[[ADR]]
// CHECK: br
  case 'l':
    s += va_arg(ap, long);
    break;

// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK-DAG: %[[NXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 8
// CHECK-DAG: store i8* %[[NXT]], i8** %ap
// CHECK-DAG: %[[ADR:[^ ]+]] = bitcast i8* %[[CUR]] to %struct.tiny*
// CHECK: br
  case 't':
    s += va_arg(ap, struct tiny).a;
    break;

// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK-DAG: %[[NXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 16
// CHECK-DAG: store i8* %[[NXT]], i8** %ap
// CHECK-DAG: %[[ADR:[^ ]+]] = bitcast i8* %[[CUR]] to %struct.small*
// CHECK: br
  case 's':
    s += *va_arg(ap, struct small).a;
    break;

// CHECK: %[[CUR:[^ ]+]] = load i8*, i8** %ap
// CHECK-DAG: %[[NXT:[^ ]+]] = getelementptr inbounds i8, i8* %[[CUR]], i64 8
// CHECK-DAG: store i8* %[[NXT]], i8** %ap
// CHECK-DAG: %[[IND:[^ ]+]] = bitcast i8* %[[CUR]] to %struct.medium**
// CHECK-DAG: %[[ADR:[^ ]+]] = load %struct.medium*, %struct.medium** %[[IND]]
// CHECK: br
  case 'm':
    s += *va_arg(ap, struct medium).a;
    break;
  }
  return s;
}
