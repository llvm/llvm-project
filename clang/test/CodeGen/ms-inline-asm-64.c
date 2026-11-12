// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -Wno-strict-prototypes -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1(void) {
  int var = 10;
  __asm mov rax, offset var ; rax = address of myvar
// CHECK: t1
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov rax, $0
// CHECK-SAME: "r,~{rax},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}})
}

void t2(void) {
  int var = 10;
  __asm mov qword ptr [eax], offset var
// CHECK: t2
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov qword ptr [eax], $0
// CHECK-SAME: "r,~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}})
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
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.t3_type) %{{.*}})
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
// CHECK-SAME: "*m,~{eax},~{ebx},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.t3_type) %{{.*}})
}

void bar() {}
static void (*fptr)();

void t5(void) {
  __asm {
    call bar
    jmp bar
    call fptr
    jmp fptr
  }
  // CHECK: t5
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: call ${0:P}
  // CHECK-SAME: jmp ${1:P}
  // CHECK-SAME: call $2
  // CHECK-SAME: jmp $3
  // CHECK-SAME: "*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(void (...)) @bar, ptr elementtype(void (...)) @bar, ptr elementtype(ptr) @fptr, ptr elementtype(ptr) @fptr)
}

void t47(void) {
  // CHECK-LABEL: define{{.*}} void @t47
  int arr[1000];
  __asm movdir64b rax, zmmword ptr [arr]
  // CHECK: call void asm sideeffect inteldialect "movdir64b rax, zmmword ptr $0", "*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype([1000 x i32]) %arr)
}
