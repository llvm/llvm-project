// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 %s -emit-llvm -fzos-extensions -triple s390x-none-zos -fvisibility=hidden -o - | FileCheck %s

// Testing pragma export after decl.
void f0(void) {}
int v0;
int vd = 2;
#pragma export(f0)
#pragma export(v0)
#pragma export(vd)

// Testing pragma export before decl.
#pragma export(f1)
#pragma export(v1)
void f1(void) {}
int v1;

void f2(void);

void t0(void) { f2();}

#pragma export(f2)
void f2(void) {}

int func() {
  int local;
  int l2;
  return local+l2;
}

int local = 2;
int l2 =4;

// CHECK: @vd = hidden global i32
// CHECK: @local = hidden global i32
// CHECK: @l2 = hidden global i32
// CHECK: @v0 = global i32
// CHECK: @v1 = global i32
// CHECK: define hidden void @f0()
// CHECK: define void @f1()
// CHECK: define hidden void @t0()
// CHECK: define void @f2()
