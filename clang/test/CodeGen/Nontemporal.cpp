// Test frontend handling of nontemporal builtins.
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;
float f1, f2;
double d1, d2;
float __attribute__((vector_size(16))) vf1, vf2;
char __attribute__((vector_size(8))) vc1, vc2;
bool b1, b2;

void test_all_sizes(void)                 // CHECK-LABEL: test_all_sizes
{
  __builtin_nontemporal_store(true, &b1); // CHECK: store i8 1, ptr @b1, align 1, !nontemporal
  __builtin_nontemporal_store(b1, &b2);   // CHECK: store i8{{.*}}, align 1, !nontemporal
  __builtin_nontemporal_store(1, &uc);    // CHECK: store i8{{.*}}align 1, !nontemporal
  __builtin_nontemporal_store(1, &sc);    // CHECK: store i8{{.*}}align 1, !nontemporal
  __builtin_nontemporal_store(1, &us);    // CHECK: store i16{{.*}}align 2, !nontemporal
  __builtin_nontemporal_store(1, &ss);    // CHECK: store i16{{.*}}align 2, !nontemporal
  __builtin_nontemporal_store(1, &ui);    // CHECK: store i32{{.*}}align 4, !nontemporal
  __builtin_nontemporal_store(1, &si);    // CHECK: store i32{{.*}}align 4, !nontemporal
  __builtin_nontemporal_store(1, &ull);   // CHECK: store i64{{.*}}align 8, !nontemporal
  __builtin_nontemporal_store(1, &sll);   // CHECK: store i64{{.*}}align 8, !nontemporal
  __builtin_nontemporal_store(1.0, &f1);  // CHECK: store float{{.*}}align 4, !nontemporal
  __builtin_nontemporal_store(1.0, &d1);  // CHECK: store double{{.*}}align 8, !nontemporal
  __builtin_nontemporal_store(vf1, &vf2); // CHECK: store <4 x float>{{.*}}align 16, !nontemporal
  __builtin_nontemporal_store(vc1, &vc2); // CHECK: store <8 x i8>{{.*}}align 8, !nontemporal

  b1 = __builtin_nontemporal_load(&b2);   // CHECK: load i8{{.*}}align 1, !nontemporal
  uc = __builtin_nontemporal_load(&sc);   // CHECK: load i8{{.*}}align 1, !nontemporal
  sc = __builtin_nontemporal_load(&uc);   // CHECK: load i8{{.*}}align 1, !nontemporal
  us = __builtin_nontemporal_load(&ss);   // CHECK: load i16{{.*}}align 2, !nontemporal
  ss = __builtin_nontemporal_load(&us);   // CHECK: load i16{{.*}}align 2, !nontemporal
  ui = __builtin_nontemporal_load(&si);   // CHECK: load i32{{.*}}align 4, !nontemporal
  si = __builtin_nontemporal_load(&ui);   // CHECK: load i32{{.*}}align 4, !nontemporal
  ull = __builtin_nontemporal_load(&sll); // CHECK: load i64{{.*}}align 8, !nontemporal
  sll = __builtin_nontemporal_load(&ull); // CHECK: load i64{{.*}}align 8, !nontemporal
  f1 = __builtin_nontemporal_load(&f2);   // CHECK: load float{{.*}}align 4, !nontemporal
  d1 = __builtin_nontemporal_load(&d2);   // CHECK: load double{{.*}}align 8, !nontemporal
  vf2 = __builtin_nontemporal_load(&vf1); // CHECK: load <4 x float>{{.*}}align 16, !nontemporal
  vc2 = __builtin_nontemporal_load(&vc1); // CHECK: load <8 x i8>{{.*}}align 8, !nontemporal
}

struct S { char c[16]; };
S x;

typedef int v4si __attribute__ ((vector_size(16)));

// CHECK-LABEL: define void @_Z14test_alignmentv()
// CHECK: load <4 x i32>, ptr @x, align 1, !nontemporal
// CHECK: store <4 x i32> %1, ptr @x, align 1, !nontemporal

void test_alignment() {
 auto t =  __builtin_nontemporal_load((v4si*)x.c);
 __builtin_nontemporal_store(t, (v4si*)x.c);
}
