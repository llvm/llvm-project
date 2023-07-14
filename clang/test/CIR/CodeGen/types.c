// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cpp.cir
// RUN: FileCheck --input-file=%t.cpp.cir --check-prefix=CHECK-CPP %s

int t0(int i) { return i; }
unsigned int t1(unsigned int i) { return i; }

char t2(char i) { return i; }
unsigned char t3(unsigned char i) { return i; }

short t4(short i) { return i; }
unsigned short t5(unsigned short i) { return i; }

float t6(float i) { return i; }
double t7(double i) { return i; }
long double t10(long double i) { return i; }

void t8(void) {}

#ifdef __cplusplus
bool t9(bool b) { return b; }
#endif

// CHECK: cir.func @t0(%arg0: !s32i loc({{.*}})) -> !s32i
// CHECK: cir.func @t1(%arg0: !u32i loc({{.*}})) -> !u32i
// CHECK: cir.func @t2(%arg0: !s8i loc({{.*}})) -> !s8i
// CHECK: cir.func @t3(%arg0: !u8i loc({{.*}})) -> !u8i
// CHECK: cir.func @t4(%arg0: !s16i loc({{.*}})) -> !s16i
// CHECK: cir.func @t5(%arg0: !u16i loc({{.*}})) -> !u16i
// CHECK: cir.func @t6(%arg0: f32 loc({{.*}})) -> f32
// CHECK: cir.func @t7(%arg0: f64 loc({{.*}})) -> f64
// CHECK: cir.func @t10(%arg0: f80 loc({{.*}})) -> f80
// CHECK: cir.func @t8()

// CHECK-CPP: cir.func @_Z2t0i(%arg0: !s32i loc({{.*}})) -> !s32i
// CHECK-CPP: cir.func @_Z2t1j(%arg0: !u32i loc({{.*}})) -> !u32i
// CHECK-CPP: cir.func @_Z2t2c(%arg0: !s8i loc({{.*}})) -> !s8i
// CHECK-CPP: cir.func @_Z2t3h(%arg0: !u8i loc({{.*}})) -> !u8i
// CHECK-CPP: cir.func @_Z2t4s(%arg0: !s16i loc({{.*}})) -> !s16i
// CHECK-CPP: cir.func @_Z2t5t(%arg0: !u16i loc({{.*}})) -> !u16i
// CHECK-CPP: cir.func @_Z2t6f(%arg0: f32 loc({{.*}})) -> f32
// CHECK-CPP: cir.func @_Z2t7d(%arg0: f64 loc({{.*}})) -> f64
// CHECK-CPP: cir.func @{{.+}}t10{{.+}}(%arg0: f80 loc({{.*}})) -> f80
// CHECK-CPP: cir.func @_Z2t8v()
// CHECK-CPP: cir.func @_Z2t9b(%arg0: !cir.bool loc({{.*}})) -> !cir.bool
