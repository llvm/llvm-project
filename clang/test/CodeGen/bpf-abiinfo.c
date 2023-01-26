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
