// RUN: %clang_cc1 -triple loongarch32 -target-feature +lsx -target-feature \
// RUN:   +lasx -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang_cc1 -triple loongarch64 -target-feature +lsx -target-feature \
// RUN:   +lasx -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64

#include <stddef.h>
#include <stdint.h>

char *s1 = "1234";
// LA32: @.str{{.*}} ={{.*}} constant [5 x i8] c"1234\00", align 1
// LA64: @.str{{.*}} ={{.*}} constant [5 x i8] c"1234\00", align 1

char *s2 = "12345678abcd";
// LA32: @.str{{.*}} ={{.*}} constant [13 x i8] c"12345678abcd\00", align 1
// LA64: @.str{{.*}} ={{.*}} constant [13 x i8] c"12345678abcd\00", align 1

char *s3 = "123456789012345678901234567890ab";
// LA32: @.str{{.*}} ={{.*}} constant [33 x i8] c"1234{{.*}}ab\00", align 1
// LA64: @.str{{.*}} ={{.*}} constant [33 x i8] c"1234{{.*}}ab\00", align 1

char *s4 = "123456789012345678901234567890123456789012345678901234567890abcdef";
// LA32: @.str{{.*}} ={{.*}} constant [67 x i8] c"1234{{.*}}cdef\00", align 1
// LA64: @.str{{.*}} ={{.*}} constant [67 x i8] c"1234{{.*}}cdef\00", align 1

int8_t a;
// LA32: @a ={{.*}} global i8 0, align 1
// LA64: @a ={{.*}} global i8 0, align 1

int16_t b;
// LA32: @b ={{.*}} global i16 0, align 2
// LA64: @b ={{.*}} global i16 0, align 2

int32_t c;
// LA32: @c ={{.*}} global i32 0, align 4
// LA64: @c ={{.*}} global i32 0, align 4

int64_t d;
// LA32: @d ={{.*}} global i64 0, align 8
// LA64: @d ={{.*}} global i64 0, align 8

intptr_t e;
// LA32: @e ={{.*}} global i32 0, align 4
// LA64: @e ={{.*}} global i64 0, align 8

float f;
// LA32: @f ={{.*}} global float 0.000000e+00, align 4
// LA64: @f ={{.*}} global float 0.000000e+00, align 4

double g;
// LA32: @g ={{.*}} global double 0.000000e+00, align 8
// LA64: @g ={{.*}} global double 0.000000e+00, align 8

struct H {
  int8_t a;
};
struct H h;
// LA32: @h ={{.*}} global %struct.H zeroinitializer, align 1
// LA64: @h ={{.*}} global %struct.H zeroinitializer, align 1
