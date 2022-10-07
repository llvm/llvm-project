// RUN: %clang_cc1 -triple i686-windows-itanium -emit-llvm -o - %s | FileCheck %s

struct f1 {
  float f;
};

struct f1 return_f1(void) { while (1); }

// CHECK: define dso_local i32 @return_f1()

void receive_f1(struct f1 a0) { }

// CHECK: define dso_local void @receive_f1(float %a0.0)

struct f2 {
  float f;
  float g;
};

struct f2 return_f2(void) { while (1); }

// CHECK: define dso_local i64 @return_f2()

void receive_f2(struct f2 a0) { }

// CHECK: define dso_local void @receive_f2(float %a0.0, float %a0.1)

struct f4 {
  float f;
  float g;
  float h;
  float i;
};

struct f4 return_f4(void) { while (1); }

// CHECK: define dso_local void @return_f4(ptr noalias sret(%struct.f4) align 4 %agg.result)

void receive_f4(struct f4 a0) { }

// CHECK: define dso_local void @receive_f4(float %a0.0, float %a0.1, float %a0.2, float %a0.3)

