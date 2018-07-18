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

#pragma code_seg("another")
// Member functions
struct __declspec(code_seg("foo_four")) Foo {
  int bar3() {return 0;}
 __declspec(code_seg("foo_lala")) int bar4() {return 0;} }; int caller() {Foo f; return f.bar3() + f.bar4(); }

//CHECK: define {{.*}}bar3@Foo{{.*}} section "foo_four"
//CHECK: define {{.*}}bar4@Foo{{.*}} section "foo_lala"

// Lambdas
#pragma code_seg("something")

int __declspec(code_seg("foo")) bar1()
{
  int lala = 4;
  auto l = [=](int i) { return i+4; };
  return l(-4);
}

//CHECK: define {{.*}}bar1{{.*}} section "foo"
//CHECK: define {{.*}}lambda{{.*}}bar1{{.*}} section "something"

double __declspec(code_seg("foo")) bar2()
{
  double lala = 4.0;
  auto l = [=](double d) __declspec(code_seg("another"))  { return d+4.0; };
  return l(4.0);
}

//CHECK: define {{.*}}bar2{{.*}} section "foo"
//CHECK: define {{.*}}lambda{{.*}}bar2{{.*}} section "another"


