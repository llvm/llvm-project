// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1(void) {
  int var = 10;
  __asm mov rax, offset var ; rax = address of myvar
// CHECK: t1
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov rax, $0
// CHECK-SAME: "r,~{rax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t2(void) {
  int var = 10;
  __asm mov qword ptr [eax], offset var
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov qword ptr [eax], $0
// CHECK-SAME: "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

struct t3_type { int a, b; };

int t3(void) {
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
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(%struct.t3_type* elementtype(%struct.t3_type) %{{.*}})
}

int t4(void) {
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
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(%struct.t3_type* elementtype(%struct.t3_type) %{{.*}})
}

void bar() {}

void t5(void) {
  __asm {
    call bar
    jmp bar
  }
  // CHECK: t5
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: call qword ptr ${0:P}
  // CHECK-SAME: jmp qword ptr ${1:P}
  // CHECK-SAME: "*m,*m,~{dirflag},~{fpsr},~{flags}"(void (...)* elementtype(void (...)) bitcast (void ()* @bar to void (...)*), void (...)* elementtype(void (...)) bitcast (void ()* @bar to void (...)*))
}
