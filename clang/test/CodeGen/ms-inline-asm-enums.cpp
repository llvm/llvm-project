// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -fasm-blocks -triple i386-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace x {
  enum { A = 12 };
  struct y_t {
    enum { A = 17 };
    int r;
  } y;
}

// CHECK-LABEL: t1
void t1() {
  enum { A = 1 };
  // CHECK: call void asm
  // CHECK-SAME: mov eax, $$12
  __asm mov eax, x::A
  // CHECK-SAME: mov eax, $$17
  __asm mov eax, x::y_t::A
  // CHECK-NEXT: call void asm
  // CHECK-SAME: mov eax, $$1
  __asm {mov eax, A}
}

// CHECK-LABEL: t2
void t2() {
  enum { A = 1, B };
  // CHECK: call void asm
  // CHECK-SAME: mov eax, $$21
  __asm mov eax, (A + 9) * 2 + A
  // CHECK-SAME: mov eax, $$4
  __asm mov eax, A << 2
  // CHECK-SAME: mov eax, $$2
  __asm mov eax, B & 3
  // CHECK-SAME: mov eax, $$5
  __asm mov eax, 3 + (B & 3)
  // CHECK-SAME: mov eax, $$8
  __asm mov eax, 2 << A * B
}

// CHECK-LABEL: t3
void t3() {
  int arr[4];
  enum { A = 4, B };
  // CHECK: call void asm
  // CHECK-SAME: mov eax, [eax + $$47]
  __asm { mov eax, [(x::A + 9) + A * B + 3 + 3 + eax] }
  // CHECK-NEXT: call void asm
  // CHECK-SAME: mov eax, dword ptr $0[$$4]
  __asm { mov eax, dword ptr [arr + A] }
  // CHECK-NEXT: call void asm
  // CHECK-SAME: mov eax, dword ptr $0[$$8]
  __asm { mov eax, dword ptr A[arr + A] }
}

