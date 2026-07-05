// RUN: %clang_cc1 -emit-llvm -triple i686-pc-win32 -std=c++11 -fms-extensions -verify -o - %s | FileCheck %s
// expected-no-diagnostics

// Class member templates

#pragma code_seg(push, "something")

template <typename T>
struct __declspec(code_seg("foo_one")) ClassOne {
  int bar1(T t) { return int(t); }
  int bar2(T t);
  int bar3(T t);
};

template <typename T>
int ClassOne<T>::bar2(T t) {
  return int(t);
}

int caller1() {
  ClassOne<int> coi;
  return coi.bar1(6) + coi.bar2(3);
}

//CHECK: define {{.*}}bar1@?$ClassOne{{.*}} section "foo_one"
//CHECK: define {{.*}}bar2@?$ClassOne{{.*}} section "foo_one"


template <typename T>
struct ClassTwo {
  int bar11(T t) { return int(t); }
  int bar22(T t);
  int bar33(T t);
};

#pragma code_seg("newone")

template <typename T>
int ClassTwo<T>::bar22(T t) {
  return int(t);
}

#pragma code_seg("someother")

template <typename T>
int ClassTwo<T>::bar33(T t) {
  return int(t);
}

#pragma code_seg("yetanother")

int caller2() {
  ClassTwo<int> coi;
  return coi.bar11(6) + coi.bar22(3) + coi.bar33(44);
}

//CHECK: define {{.*}}bar11@?$ClassTwo{{.*}} section "something"
//CHECK: define {{.*}}bar22@?$ClassTwo{{.*}} section "newone"
//CHECK: define {{.*}}bar33@?$ClassTwo{{.*}} section "someother"

template<>
struct ClassOne<double>
{
  int bar44(double d) { return 1; }
};
template<>
struct  __declspec(code_seg("foo_three")) ClassOne<long>
{
  int bar55(long d) { return 1; }
};

#pragma code_seg("onemore")
int caller3() {
  ClassOne<double> d;
  ClassOne<long> l;
  return d.bar44(1.0)+l.bar55(1);
}

//CHECK: define {{.*}}bar44{{.*}} section "yetanother"
//CHECK: define {{.*}}bar55{{.*}} section "foo_three"


// Function templates
template <typename T>
int __declspec(code_seg("foo_four")) bar66(T t) { return int(t); }

// specializations do not take the segment from primary
template<>
int bar66(int i) { return 0; }

#pragma code_seg(pop)

template<>
int bar66(char c) { return 0; }

struct A1 {int i;};
template<>
int __declspec(code_seg("foo_five")) bar66(A1 a) { return a.i; }

int caller4()
{
// but instantiations do use the section from the primary
return bar66(0) + bar66(1.0) + bar66('c');
}
//CHECK: define {{.*}}bar66@H{{.*}} section "onemore"
//CHECK-NOT: define {{.*}}bar66@D{{.*}} section
//CHECK: define {{.*}}bar66@UA1{{.*}} section "foo_five"
//CHECK: define {{.*}}bar66@N{{.*}} section "foo_four"



