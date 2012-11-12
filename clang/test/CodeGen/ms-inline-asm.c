// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -O0 -fms-extensions -fenable-experimental-ms-inline-asm -w -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret void
  __asm {}
}

void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret void
  __asm nop
  __asm nop
  __asm nop
}

void t3() {
// CHECK: @t3
// CHECK: call void asm sideeffect inteldialect "nop\0A\09nop\0A\09nop", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret void
  __asm nop __asm nop __asm nop
}

void t4(void) {
// CHECK: @t4
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax", "~{ebx},~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "mov ecx, ebx", "~{ecx},~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret void
  __asm mov ebx, eax
  __asm mov ecx, ebx
}

void t5(void) {
// CHECK: @t5
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax\0A\09mov ecx, ebx", "~{ebx},~{ecx},~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret void
  __asm mov ebx, eax __asm mov ecx, ebx
}

void t6(void) {
  __asm int 0x2c
// CHECK: t6
// CHECK: call void asm sideeffect inteldialect "int $$0x2c", "~{dirflag},~{fpsr},~{flags}"() nounwind
}

void t7() {
  __asm {
    int 0x2c ; } asm comments are fun! }{
  }
  __asm {}
// CHECK: t7
// CHECK: call void asm sideeffect inteldialect "int $$0x2c", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"() nounwind
}

int t8() {
  __asm int 4 ; } comments for single-line asm
  __asm {}
  __asm int 4
  return 10;
// CHECK: t8
// CHECK: call void asm sideeffect inteldialect "int $$4", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: call void asm sideeffect inteldialect "int $$4", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK: ret i32 10
}

void t9() {
  __asm {
    push ebx
    mov ebx, 0x07
    pop ebx
  }
// CHECK: t9
// CHECK: call void asm sideeffect inteldialect "push ebx\0A\09mov ebx, $$0x07\0A\09pop ebx", "~{ebx},~{dirflag},~{fpsr},~{flags}"() nounwind
}

unsigned t10(void) {
  unsigned i = 1, j;
  __asm {
    mov eax, i
    mov j, eax
  }
  return j;
// CHECK: t10
// CHECK: [[I:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: [[J:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: store i32 1, i32* [[I]], align 4
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $1\0A\09mov dword ptr $0, eax", "=*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}, i32* %{{.*}}) nounwind
// CHECK: [[RET:%[a-zA-Z0-9]+]] = load i32* [[J]], align 4
// CHECK: ret i32 [[RET]]
}

void t11(void) {
  __asm mov eax, 1
// CHECK: t11
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"() nounwind
}

unsigned t12(void) {
  unsigned i = 1, j, l = 1, m;
  __asm {
    mov eax, i
    mov j, eax
    mov eax, l
    mov m, eax
  }
  return j + m;
// CHECK: t12
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $2\0A\09mov dword ptr $0, eax\0A\09mov eax, dword ptr $3\0A\09mov dword ptr $1, eax", "=*m,=*m,*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}) nounwind
}

void t13() {
  char i = 1;
  short j = 2;
  __asm movzx eax, i
  __asm movzx eax, j
// CHECK: t13
// CHECK: call void asm sideeffect inteldialect "movzx eax, byte ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i8* %{{.*}}) nounwind
// CHECK: call void asm sideeffect inteldialect "movzx eax, word ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i16* %{{.*}}) nounwind
}

void t14() {
  unsigned i = 1, j = 2;
  __asm {
    .if 1
    mov eax, i
    .else
    mov ebx, j
    .endif
  }
// CHECK: t14
// CHECK: call void asm sideeffect inteldialect ".if 1\0A\09mov eax, dword ptr $0\0A\09.else\0A\09mov ebx, j\0A\09.endif", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
}

void t15() {
  int var = 10;
  __asm mov eax, var        ; eax = 10
  __asm mov eax, offset var ; eax = address of myvar
// CHECK: t15
// CHECK: call void asm sideeffect inteldialect "mov eax, dword ptr $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
}

void t16() {
  int var = 10;
  __asm mov [eax], offset var
// CHECK: t16
// CHECK: call void asm sideeffect inteldialect "mov [eax], $0", "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}}) nounwind
}

void t17() {
  __asm _emit 0x4A
  __asm _emit 0x43
  __asm _emit 0x4B
// CHECK: t17
// CHECK:  call void asm sideeffect inteldialect ".byte 0x4A", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK:  call void asm sideeffect inteldialect ".byte 0x43", "~{dirflag},~{fpsr},~{flags}"() nounwind
// CHECK:  call void asm sideeffect inteldialect ".byte 0x4B", "~{dirflag},~{fpsr},~{flags}"() nounwind
}

struct t18_type { int a, b; };

int t18() {
  struct t18_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     mov eax, [ebx].0
     mov [ebx].4, ecx
  }
  return foo.b;
// CHECK: t18
// CHECK: call void asm sideeffect inteldialect "lea ebx, foo\0A\09mov eax, [ebx].0\0A\09mov [ebx].4, ecx", "~{eax},~{dirflag},~{fpsr},~{flags}"() nounwind
}

int t19() {
  struct t18_type foo;
  foo.a = 1;
  foo.b = 2;
  __asm {
     lea ebx, foo
     mov eax, [ebx].foo.a
     mov [ebx].foo.b, ecx
  }
  return foo.b;
// CHECK: t19
// CHECK: call void asm sideeffect inteldialect "lea ebx, foo\0A\09mov eax, [ebx].0\0A\09mov [ebx].4, ecx", "~{eax},~{dirflag},~{fpsr},~{flags}"() nounwind
}

void t20() {
  int foo;
  __asm mov eax, TYPE foo
// CHECK: t20
// CHECK: call void asm sideeffect inteldialect "mov eax, $$4", "~{eax},~{dirflag},~{fpsr},~{flags}"() nounwind
}
