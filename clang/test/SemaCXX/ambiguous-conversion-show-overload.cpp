// RUN: not %clang_cc1 -fsyntax-only -fshow-overloads=best -fno-caret-diagnostics %s 2>&1 | FileCheck %s
struct S {
  S(void*);
  S(char*);
  S(unsigned char*);
  S(signed char*);
  S(unsigned short*);
  S(signed short*);
  S(unsigned int*);
  S(signed int*);
};
void f(const S& s);

// First call to f emits all candidates.  Second call emits just the first 4.
void g() { f(0); }
// CHECK: {{conversion from 'int' to 'const S' is ambiguous}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}

void h() { f(0); }
// CHECK: {{conversion from 'int' to 'const S' is ambiguous}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{candidate constructor}}
// CHECK-NEXT: {{remaining 4 candidates omitted; pass -fshow-overloads=all to show them}}
