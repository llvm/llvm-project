// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-linux-gnu | FileCheck %s

// Test address-of overloading logic
int test5(int);
template <typename T>
T test5(T) __attribute__((enable_if(1, "better than non-template")));

// CHECK: @_Z5test5IiEUa9enable_ifIXLi1EEET_S0_
int (*Ptr)(int) = &test5;

// Test itanium mangling for attribute enable_if

// CHECK: _Z5test1Ua9enable_ifIXeqfL0p_Li1EEEi
void test1(int i) __attribute__((enable_if(i == 1, ""))) {}

void ext();
// CHECK: _Z5test2Ua9enable_ifIXneadL_Z3extvELi0EEEi
void test2(int i) __attribute__((enable_if(&ext != 0, ""))) {}

// CHECK: _Z5test3Ua9enable_ifIXeqfL0p_Li1EEXeqfL0p0_Li2EEEii
void test3(int i, int j) __attribute__((enable_if(i == 1, ""), enable_if(j == 2, ""))) {}

// CHECK: _ZN5test4IdE1fEUa9enable_ifIXeqfL0p_Li1EEXeqfL0p0_Li2EEEi
template <typename T>
class test4 {
  virtual void f(int i, int j) __attribute__((enable_if(i == 1, ""))) __attribute__((enable_if(j == 2, "")));
};

template class test4<double>;
