// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -fms-extensions -verify -o - %s | FileCheck %s
// expected-no-diagnostics

// Simple case

int __declspec(code_seg("foo_one")) bar_one() { return 1; }
//CHECK: define {{.*}}bar_one{{.*}} section "foo_one"

// Simple case - explicit attribute used over pragma
#pragma code_seg("foo_two")
int __declspec(code_seg("foo_three")) bar2() { return 2; }
//CHECK: define {{.*}}bar2{{.*}} section "foo_three"

// Check that attribute on one function doesn't affect another
int another1() { return 1001; }
//CHECK: define {{.*}}another1{{.*}} section "foo_two"

// Member functions

struct __declspec(code_seg("foo_four")) Foo {
  int bar3() {return 0;}
  int bar4();
  int __declspec(code_seg("foo_six")) bar6() { return 6; }
  int bar7() { return 7; }
  struct Inner {
    int bar5() { return 5; }
  } z;
  virtual int baz1() { return 1; }
};

struct __declspec(code_seg("foo_four")) FooTwo : Foo {
  int baz1() { return 20; }
};

int caller1() {
  Foo f; return f.bar3();
}

//CHECK: define {{.*}}bar3@Foo{{.*}} section "foo_four"
int Foo::bar4() { return 4; }
//CHECK: define {{.*}}bar4@Foo{{.*}} section "foo_four"

#pragma code_seg("someother")

int caller2() {
  Foo f;
  Foo *fp = new FooTwo;
  return f.z.bar5() + f.bar6() + f.bar7() + fp->baz1();
}
// TBD: MS Compiler and Docs do not match for nested routines
// Doc says:      define {{.*}}bar5@Inner@Foo{{.*}} section "foo_four"
// Compiler says: define {{.*}}bar5@Inner@Foo{{.*}} section "foo_two"
//CHECK: define {{.*}}bar6@Foo{{.*}} section "foo_six"
//CHECK: define {{.*}}bar7@Foo{{.*}} section "foo_four"
// Check that code_seg active at class declaration is not used on member
// declared outside class when it is not active.

#pragma code_seg(push,"AnotherSeg")

struct FooThree {
  int bar8();
  int bar9() { return 9; }
};

#pragma code_seg(pop)


int FooThree::bar8() {return 0;}

int caller3()
{
  FooThree f;
  return f.bar8() + f.bar9();
}

//CHECK: define {{.*}}bar8@FooThree{{.*}} section "someother"
//CHECK: define {{.*}}bar9@FooThree{{.*}} section "AnotherSeg"

struct NonTrivialCopy {
  NonTrivialCopy();
  NonTrivialCopy(const NonTrivialCopy&);
  ~NonTrivialCopy();
};

// check the section for compiler-generated function with declspec.

struct __declspec(code_seg("foo_seven")) FooFour {
  FooFour() {}
  int __declspec(code_seg("foo_eight")) bar10(int t) { return t; }
  NonTrivialCopy f;
};

//CHECK: define {{.*}}0FooFour@@QAE@ABU0@@Z{{.*}} section "foo_seven"
// check the section for compiler-generated function with no declspec.

struct FooFive {
  FooFive() {}
  int __declspec(code_seg("foo_nine")) bar11(int t) { return t; }
  NonTrivialCopy f;
};

//CHECK: define {{.*}}0FooFive@@QAE@ABU0@@Z{{.*}} section "someother"

#pragma code_seg("YetAnother")
int caller4()
{
  FooFour z1;
  FooFour z2 = z1;
  FooFive y1;
  FooFive y2 = y1;
  return z2.bar10(0) + y2.bar11(1);
}

//CHECK: define {{.*}}bar10@FooFour{{.*}} section "foo_eight"
//CHECK: define {{.*}}bar11@FooFive{{.*}} section "foo_nine"

struct FooSix {
  #pragma code_seg("foo_ten")
  int bar12() { return 12; }
  #pragma code_seg("foo_eleven")
  int bar13() { return 13; }
};

int bar14() { return 14; }
//CHECK: define {{.*}}bar14{{.*}} section "foo_eleven"

int caller5()
{
  FooSix fsix;
  return fsix.bar12() + fsix.bar13();
}

//CHECK: define {{.*}}bar12@FooSix{{.*}} section "foo_ten"
//CHECK: define {{.*}}bar13@FooSix{{.*}} section "foo_eleven"
//CHECK: define {{.*}}baz1@FooTwo{{.*}} section "foo_four"
