// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);

int f1(int arg)  { return arg; }; 

int f2(float arg) { return int(arg); }; 

typedef int (*fp1)(int); 

typedef int (*fp2)(float); 

struct A {
  operator fp1() { return f1; }
  operator fp2() { return f2; } 
} a;


// Test for function reference.
typedef int (&fr1)(int); 
typedef int (&fr2)(float); 

struct B {
  operator fr1() { return f1; }
  operator fr2() { return f2; } 
} b;

int main()
{
 int i = a(10); // Calls f1 via pointer returned from conversion function
 printf("i = %d\n", i);

 int j = b(20); // Calls f1 via pointer returned from conversion function
 printf("j = %d\n", j);
 return 0;
}

// CHECK-LP64: callq __ZN1AcvPFiiEEv
// CHECK-LP64: callq __ZN1BcvRFiiEEv

// CHECK-LP32: calll L__ZN1AcvPFiiEEv
// CHECK-LP32: calll L__ZN1BcvRFiiEEv

