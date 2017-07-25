// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -fasm-blocks -emit-llvm -o - | FileCheck %s
namespace x {
enum { A = 12 };
struct y_t {
	enum { A = 17 };
	int r;
} y;
}
// CHECK-LABEL: x86_enum_only
void x86_enum_only(){
  const int a = 0;
  // CHECK-NOT: mov eax, [$$0]
  // Other constant type folding is currently unwanted.
  __asm mov eax, [a]
  }

// CHECK-LABEL: x86_enum_namespaces
void x86_enum_namespaces() {
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

// CHECK-LABEL: x86_enum_arithmethic
void x86_enum_arithmethic() {
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

// CHECK-LABEL: x86_enum_mem
void x86_enum_mem() {
  int arr[4];
  enum { A = 4, B };
  // CHECK: call void asm
  // CHECK-SAME: mov eax, [($$12 + $$9) + $$4 * $$5 + $$3 + $$3 + eax]
  __asm { mov eax, [(x::A + 9) + A * B + 3 + 3 + eax] }
  // CHECK-NEXT: call void asm
  // CHECK-SAME: mov eax, dword ptr $$4$0
  __asm { mov eax, dword ptr [arr + A] }
  // CHECK-NEXT: call void asm
  // CHECK-SAME: mov eax, dword ptr $$8$0
  __asm { mov eax, dword ptr A[arr + A] }
}
