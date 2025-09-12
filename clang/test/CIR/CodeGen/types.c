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

// CHECK: cir.func dso_local @t0(%arg0: !s32i loc({{.*}})) -> !s32i
// CHECK: cir.func dso_local @t1(%arg0: !u32i loc({{.*}})) -> !u32i
// CHECK: cir.func dso_local @t2(%arg0: !s8i loc({{.*}})) -> !s8i
// CHECK: cir.func dso_local @t3(%arg0: !u8i loc({{.*}})) -> !u8i
// CHECK: cir.func dso_local @t4(%arg0: !s16i loc({{.*}})) -> !s16i
// CHECK: cir.func dso_local @t5(%arg0: !u16i loc({{.*}})) -> !u16i
// CHECK: cir.func dso_local @t6(%arg0: !cir.float loc({{.*}})) -> !cir.float
// CHECK: cir.func dso_local @t7(%arg0: !cir.double loc({{.*}})) -> !cir.double
// CHECK: cir.func dso_local @t10(%arg0: !cir.long_double<!cir.f80> loc({{.*}})) -> !cir.long_double<!cir.f80>
// CHECK: cir.func dso_local @t8()

// CHECK-CPP: cir.func dso_local @_Z2t0i(%arg0: !s32i loc({{.*}})) -> !s32i
// CHECK-CPP: cir.func dso_local @_Z2t1j(%arg0: !u32i loc({{.*}})) -> !u32i
// CHECK-CPP: cir.func dso_local @_Z2t2c(%arg0: !s8i loc({{.*}})) -> !s8i
// CHECK-CPP: cir.func dso_local @_Z2t3h(%arg0: !u8i loc({{.*}})) -> !u8i
// CHECK-CPP: cir.func dso_local @_Z2t4s(%arg0: !s16i loc({{.*}})) -> !s16i
// CHECK-CPP: cir.func dso_local @_Z2t5t(%arg0: !u16i loc({{.*}})) -> !u16i
// CHECK-CPP: cir.func dso_local @_Z2t6f(%arg0: !cir.float loc({{.*}})) -> !cir.float
// CHECK-CPP: cir.func dso_local @_Z2t7d(%arg0: !cir.double loc({{.*}})) -> !cir.double
// CHECK-CPP: cir.func dso_local @{{.+}}t10{{.+}}(%arg0: !cir.long_double<!cir.f80> loc({{.*}})) -> !cir.long_double<!cir.f80>
// CHECK-CPP: cir.func dso_local @_Z2t8v()
// CHECK-CPP: cir.func dso_local @_Z2t9b(%arg0: !cir.bool loc({{.*}})) -> !cir.bool
