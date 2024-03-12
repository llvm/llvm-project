// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -O2 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

_Bool bar_bool(void);
unsigned char bar_char(void);
short bar_short(void);
int bar_int(void);

int foo_bool(void) {
        if (bar_bool() != 1) return 0; else return 1;
}
// CHECK: %call = call i1 @bar_bool()
int foo_char(void) {
        if (bar_char() != 10) return 0; else return 1;
}
// CHECK: %call = call i8 @bar_char()
int foo_short(void) {
        if (bar_short() != 10) return 0; else return 1;
}
// CHECK: %call = call i16 @bar_short()
int foo_int(void) {
        if (bar_int() != 10) return 0; else return 1;
}
// CHECK: %call = call i32 @bar_int()

void sprog1(short, int, int);
void mprog1() {
  sprog1(-3, 4, -5);
// CHECK: call void @sprog1(i16 noundef signext -3, i32 noundef signext 4, i32 noundef signext -5)
}
void mprog2(long a, long b) {
  sprog1(a, b, b);
// CHECK: call void @sprog1(i16 noundef signext %{{[0-9a-z]+}}, i32 noundef signext %{{[0-9a-z]+}}, i32 noundef signext %{{[0-9a-z]+}})
}
