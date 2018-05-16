// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-pc-win32
// expected-no-diagnostics

// Non-Member Function Overloading is involved

int __declspec(code_seg("foo_one")) bar_one(int) { return 1; }
//CHECK: define {{.*}}bar_one{{.*}} section "foo_one"
int __declspec(code_seg("foo_two")) bar_one(int,float) { return 11; }
//CHECK: define {{.*}}bar_one{{.*}} section "foo_two"
int __declspec(code_seg("foo_three")) bar_one(float) { return 12; }
//CHECK: define {{.*}}bar_one{{.*}} section "foo_three"

// virtual function overloading is involved

struct __declspec(code_seg("my_one")) Base3 {
  virtual int barA(int) { return 1; }
  virtual int barA(int,float) { return 2; }
  virtual int barA(float) { return 3; }

  virtual void __declspec(code_seg("my_two")) barB(int) { }
  virtual void  __declspec(code_seg("my_three")) barB(float) { }
  virtual void __declspec(code_seg("my_four")) barB(int, float) { }

};

//CHECK: define {{.*}}barA@Base3{{.*}} section "my_one"
//CHECK: define {{.*}}barA@Base3{{.*}} section "my_one"
//CHECK: define {{.*}}barA@Base3{{.*}} section "my_one"
//CHECK: define {{.*}}barB@Base3{{.*}} section "my_two"
//CHECK: define {{.*}}barB@Base3{{.*}} section "my_three"
//CHECK: define {{.*}}barB@Base3{{.*}} section "my_four"
