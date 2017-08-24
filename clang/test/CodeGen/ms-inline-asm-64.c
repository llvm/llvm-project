// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1() {
  int var = 10;
  __asm mov rax, offset var ; rax = address of myvar
// CHECK: t1
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov rax, $0
// CHECK-SAME: "r,~{rax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t2() {
  int var = 10;
  __asm mov [eax], offset var
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov [eax], $0
// CHECK-SAME: "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

struct t3_type { int a, b; };

int t3() {
  struct t3_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     mov eax, [ebx].0
     mov [ebx].4, ecx
  }
  return foo.b;
// CHECK: t3
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: lea ebx, $0
// CHECK-SAME: mov eax, [ebx]
// CHECK-SAME: mov [ebx + $$4], ecx
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(%struct.t3_type* %{{.*}})
}

int t4() {
  struct t3_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     {
       mov eax, [ebx].foo.a
     }
     mov [ebx].foo.b, ecx
  }
  return foo.b;
// CHECK: t4
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: lea ebx, $0
// CHECK-SAME: mov eax, [ebx]
// CHECK-SAME: mov [ebx + $$4], ecx
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(%struct.t3_type* %{{.*}})
}
