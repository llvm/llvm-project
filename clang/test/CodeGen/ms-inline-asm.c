// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm {}
}

void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect inteldialect "nop\0A\09nop\0A\09nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm nop
  __asm nop
  __asm nop
}

void t3() {
// CHECK: @t3
// CHECK: call void asm sideeffect inteldialect "nop\0A\09nop\0A\09nop", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm nop __asm nop __asm nop
}

void t4(void) {
// CHECK: @t4
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax\0A\09mov ecx, ebx", "~{ebx},~{ecx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm mov ebx, eax
  __asm mov ecx, ebx
}

void t5(void) {
// CHECK: @t5
// CHECK: call void asm sideeffect inteldialect "mov ebx, eax\0A\09mov ecx, ebx", "~{ebx},~{ecx},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm mov ebx, eax __asm mov ecx, ebx
}

void t6(void) {
  __asm int 0x2c
// CHECK: t6
// CHECK: call void asm sideeffect inteldialect "int $$44", "~{dirflag},~{fpsr},~{flags}"()
}

void t7() {
  __asm {
    int 0x2cU ; } asm comments are fun! }{
  }
  __asm {
    {
      int 0x2c ; } asm comments are fun! }{
    }
  }
  __asm {}
  __asm {
    ;
    ; label
    mov eax, ebx
  }
// CHECK: t7
// CHECK: call void asm sideeffect inteldialect "int $$44", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "", "~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void asm sideeffect inteldialect "mov eax, ebx", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

int t8() {
  __asm int 4 ; } comments for single-line asm
  __asm {}
  __asm { int 5}
  __asm int 6
  __asm int 7
  __asm { 
    int 8
  }
  return 10;
// CHECK: t8
// CHECK: call i32 asm sideeffect inteldialect "int $$4", "={eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call i32 asm sideeffect inteldialect "", "={eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call i32 asm sideeffect inteldialect "int $$5", "={eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call i32 asm sideeffect inteldialect "int $$6\0A\09int $$7", "={eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call i32 asm sideeffect inteldialect "int $$8", "={eax},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret i32 10
}

void t9() {
  __asm {
    push ebx
    { mov ebx, 0x07 }
    __asm { pop ebx }
  }
// CHECK: t9
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: push ebx
// CHECK-SAME: mov ebx, $$7
// CHECK-SAME: pop ebx
// CHECK-SAME: "~{ebx},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

unsigned t10(void) {
  unsigned i = 1, j;
  __asm {
    mov eax, i
    mov j, eax
  }
  return j;
// CHECK: t10
// CHECK: [[r:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: [[I:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: [[J:%[a-zA-Z0-9]+]] = alloca i32, align 4
// CHECK: store i32 1, i32* [[I]], align 4
// CHECK: call i32 asm sideeffect inteldialect
// CHECK-SAME: mov eax, $2
// CHECK-SAME: mov $0, eax
// CHECK-SAME: "=*m,=&{eax},*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}})
// CHECK: [[RET:%[a-zA-Z0-9]+]] = load i32, i32* [[J]], align 4
// CHECK: ret i32 [[RET]]
}

void t11(void) {
  __asm mov eax, 1
// CHECK: t11
// CHECK: call void asm sideeffect inteldialect "mov eax, $$1", "~{eax},~{dirflag},~{fpsr},~{flags}"()
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
// CHECK: call i32 asm sideeffect inteldialect
// CHECK-SAME: mov eax, $3
// CHECK-SAME: mov $0, eax
// CHECK-SAME: mov eax, $4
// CHECK-SAME: mov $1, eax
// CHECK-SAME: "=*m,=*m,=&{eax},*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}})
}

void t13() {
  char i = 1;
  short j = 2;
  __asm movzx eax, i
  __asm movzx eax, j
// CHECK-LABEL: define{{.*}} void @t13()
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: movzx eax, byte ptr $0
// CHECK-SAME: movzx eax, word ptr $1
// CHECK-SAME: "*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i8* elementtype(i8) %{{.*}}i, i16* elementtype(i16) %{{.*}}j)
}

void t13_brac() {
  char i = 1;
  short j = 2;
  __asm movzx eax, [i]
  __asm movzx eax, [j]
// CHECK-LABEL: define{{.*}} void @t13_brac()
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: movzx eax, byte ptr $0
// CHECK-SAME: movzx eax, word ptr $1
// CHECK-SAME: "*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i8* elementtype(i8) %{{.*}}i, i16* elementtype(i16) %{{.*}}j)
}

void t14() {
  unsigned i = 1, j = 2;
  __asm {
    .if 1
    { mov eax, i }
    .else
    mov ebx, j
    .endif
  }
// CHECK: t14
// CHECK: call void asm sideeffect inteldialect ".if 1\0A\09mov eax, $0\0A\09.else\0A\09mov ebx, j\0A\09.endif", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
}

int gvar = 10;
void t15() {
// CHECK: t15
  int lvar = 10;
  __asm mov eax, lvar        ; eax = 10
// CHECK: mov eax, $0
  __asm mov eax, offset lvar ; eax = address of lvar
// CHECK: mov eax, $1
  __asm mov eax, offset gvar ; eax = address of gvar
// CHECK: mov eax, $2
  __asm mov eax, offset gvar+1 ; eax = 1 + address of gvar
// CHECK: mov eax, $3 + $$1
  __asm mov eax, 1+offset gvar ; eax = 1 + address of gvar
// CHECK: mov eax, $4 + $$1
  __asm mov eax, 1+offset gvar+1 ; eax = 2 + address of gvar
// CHECK: mov eax, $5 + $$2
// CHECK: "*m,r,i,i,i,i,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* %{{.*}}, i32* @{{.*}}, i32* @{{.*}}, i32* @{{.*}}, i32* @{{.*}})
}

void t16() {
  int var = 10;
  __asm mov dword ptr [eax], offset var
// CHECK: t16
// CHECK: call void asm sideeffect inteldialect "mov dword ptr [eax], $0", "r,~{dirflag},~{fpsr},~{flags}"(i32* %{{.*}})
}

void t17() {
// CHECK: t17
  __asm _emit 0x4A
// CHECK: .byte 0x4A
  __asm _emit 0x43L
// CHECK: .byte 0x43L
  __asm _emit 0x4B
// CHECK: .byte 0x4B
  __asm _EMIT 0x4B
// CHECK: .byte 0x4B
// CHECK:  "~{dirflag},~{fpsr},~{flags}"()
}

void t20() {
// CHECK: t20
  char bar;
  int foo;
  char _bar[2];
  int _foo[4];

  __asm mov eax, LENGTH foo
// CHECK: mov eax, $$1
  __asm mov eax, LENGTH bar
// CHECK: mov eax, $$1
  __asm mov eax, LENGTH _foo
// CHECK: mov eax, $$4
  __asm mov eax, LENGTH _bar
// CHECK: mov eax, $$2
  __asm mov eax, [eax + LENGTH foo * 4]
// CHECK: mov eax, [eax + $$4]

  __asm mov eax, TYPE foo
// CHECK: mov eax, $$4
  __asm mov eax, TYPE bar
// CHECK: mov eax, $$1
  __asm mov eax, TYPE _foo
// CHECK: mov eax, $$4
  __asm mov eax, TYPE _bar
// CHECK: mov eax, $$1
  __asm mov eax, [eax + TYPE foo * 4]
// CHECK: mov eax, [eax + $$16]

  __asm mov eax, SIZE foo
// CHECK: mov eax, $$4
  __asm mov eax, SIZE bar
// CHECK: mov eax, $$1
  __asm mov eax, SIZE _foo
// CHECK: mov eax, $$16
  __asm mov eax, [eax + SIZE _foo * 4]
// CHECK: mov eax, [eax + $$64]
  __asm mov eax, SIZE _bar
// CHECK: mov eax, $$2
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()

}

void t21() {
  __asm {
    __asm push ebx
    __asm mov ebx, 07H
    __asm pop ebx
  }
// CHECK: t21
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: push ebx
// CHECK-SAME: mov ebx, $$7
// CHECK-SAME: pop ebx
// CHECK-SAME: "~{ebx},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

extern void t22_helper(int x);
void t22() {
  int x = 0;
  __asm {
    __asm push ebx
    __asm mov ebx, esp
  }
  t22_helper(x);
  __asm {
    __asm mov esp, ebx
    __asm pop ebx
  }
// CHECK: t22
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: push ebx
// CHECK-SAME: mov ebx, esp
// CHECK-SAME: "~{ebx},~{esp},~{dirflag},~{fpsr},~{flags}"()
// CHECK: call void @t22_helper
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov esp, ebx
// CHECK-SAME: pop ebx
// CHECK-SAME: "~{ebx},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

void t23() {
  __asm {
  the_label:
  }
// CHECK: t23
// CHECK: call void asm sideeffect inteldialect "{{.*}}__MSASMLABEL_.${:uid}__the_label:", "~{dirflag},~{fpsr},~{flags}"()
}

void t24_helper(void) {}
void t24() {
  __asm call t24_helper
// CHECK: t24
// CHECK: call void asm sideeffect inteldialect "call dword ptr ${0:P}", "*m,~{dirflag},~{fpsr},~{flags}"(void ()* elementtype(void ()) @t24_helper)
}

void t25() {
// CHECK: t25
  __asm mov eax, 0ffffffffh
// CHECK: mov eax, $$4294967295
  __asm mov eax, 0fhU
// CHECK: mov eax, $$15
  __asm mov eax, 0a2h
// CHECK: mov eax, $$162
  __asm mov eax, 10100010b
// CHECK: mov eax, $$162
  __asm mov eax, 10100010BU
// CHECK: mov eax, $$162
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t26() {
// CHECK: t26
  __asm pushad
// CHECK: pushad
  __asm mov eax, 0
// CHECK: mov eax, $$0
  __asm __emit 0fh
// CHECK: .byte 0fh
  __asm __emit 0a2h
// CHECK: .byte 0a2h
  __asm __EMIT 0a2h
// CHECK: .byte 0a2h
  __asm popad
// CHECK: popad
// CHECK: "~{eax},~{ebp},~{ebx},~{ecx},~{edi},~{edx},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

void t27() {
  __asm mov eax, fs:[0h]
// CHECK: t27
// CHECK: call void asm sideeffect inteldialect "mov eax, fs:[$$0]", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t28() {
// CHECK: t28
  __asm align 8
// CHECK: .align 3
  __asm align 16;
// CHECK: .align 4
  __asm align 128;
// CHECK: .align 7
  __asm ALIGN 256;
// CHECK: .align 8
// CHECK: "~{dirflag},~{fpsr},~{flags}"()
}

void t29() {
// CHECK: t29
  int arr[2] = {0, 0};
  int olen = 0, osize = 0, otype = 0;
  __asm mov olen, LENGTH arr
// CHECK: mov dword ptr $0, $$2
  __asm mov osize, SIZE arr
// CHECK: mov dword ptr $1, $$8
  __asm mov otype, TYPE arr
// CHECK: mov dword ptr $2, $$4
// CHECK: "=*m,=*m,=*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}})
}

int results[2] = {13, 37};
int *t30()
// CHECK: t30
{
  int *res;
  __asm lea edi, results
// CHECK: lea edi, $2
  __asm mov res, edi
// CHECK: mov $0, edi
  return res;
// CHECK: "=*m,={eax},*m,~{edi},~{dirflag},~{fpsr},~{flags}"(i32** elementtype(i32*) %{{.*}}, [2 x i32]* elementtype([2 x i32]) @{{.*}})
}

void t31() {
// CHECK: t31
  __asm pushad
// CHECK: pushad
  __asm popad
// CHECK: popad
// CHECK: "~{eax},~{ebp},~{ebx},~{ecx},~{edi},~{edx},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"()
}

void t32() {
// CHECK: t32
  int i;
  __asm mov eax, i
// CHECK: mov eax, $0
  __asm mov eax, dword ptr i
// CHECK: mov eax, dword ptr $1
  __asm mov ax, word ptr i
// CHECK: mov ax, word ptr $2
  __asm mov al, byte ptr i
// CHECK: mov al, byte ptr $3
// CHECK: "*m,*m,*m,*m,~{al},~{ax},~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}})
}

void t33() {
// CHECK: t33
  int i;
  __asm mov eax, [i]
// CHECK: mov eax, $0
  __asm mov eax, dword ptr [i]
// CHECK: mov eax, dword ptr $1
  __asm mov ax, word ptr [i]
// CHECK: mov ax, word ptr $2
  __asm mov al, byte ptr [i]
// CHECK: mov al, byte ptr $3
// CHECK: "*m,*m,*m,*m,~{al},~{ax},~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}}, i32* elementtype(i32) %{{.*}})
}

void t34() {
// CHECK: t34
  __asm prefetchnta 64[eax]
// CHECK: prefetchnta [eax + $$64]
  __asm mov eax, dword ptr 4[eax]
// CHECK: mov eax, dword ptr [eax + $$4]
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t35() {
// CHECK: t35
  __asm prefetchnta [eax + (200*64)]
// CHECK: prefetchnta [eax + $$12800]
  __asm mov eax, dword ptr [eax + (200*64)]
// CHECK: mov eax, dword ptr [eax + $$12800]
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t36() {
// CHECK: t36
  int arr[4];
  // Work around PR20368: These should be single line blocks
  __asm { mov eax, 4[arr] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4[arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 8[arr + 4 + 32*2 - 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$72]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 12[4 + arr] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$16]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4[4 + arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$12]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4[64 + arr + (2*32)] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$132]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4[64 + arr - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [arr + 4 + 32*2 - 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$64]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [4 + arr] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [4 + arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [64 + arr + (2*32)] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$128]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, [64 + arr - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
}

void t37() {
// CHECK: t37
  __asm mov eax, 4 + 8
// CHECK: mov eax, $$12
  __asm mov eax, 4 + 8 * 16
// CHECK: mov eax, $$132
  __asm mov eax, -4 + 8 * 16
// CHECK: mov eax, $$124
  __asm mov eax, (4 + 4) * 16
// CHECK: mov eax, $$128
  __asm mov eax, 4 + 8 * -16
// CHECK: mov eax, $$-124
  __asm mov eax, 4 + 16 / -8
// CHECK: mov eax, $$2
  __asm mov eax, (16 + 16) / -8
// CHECK: mov eax, $$-4
  __asm mov eax, ~15
// CHECK: mov eax, $$-16
  __asm mov eax, 6 ^ 3
// CHECK: mov eax, $$5
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t38() {
// CHECK: t38
  int arr[4];
  // Work around PR20368: These should be single line blocks
  __asm { mov eax, 4+4[arr] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, (4+4)[arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$12]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 8*2[arr + 4 + 32*2 - 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$80]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 12+20[4 + arr] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$36]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4*16+4[4 + arr + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$76]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4*4[64 + arr + (2*32)] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$144]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 4*(4-2)[64 + arr - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
  __asm { mov eax, 32*(4-2)[arr - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"([4 x i32]* elementtype([4 x i32]) %{{.*}})
}

void cpuid() {
  __asm cpuid
// CHECK-LABEL: define{{.*}} void @cpuid
// CHECK: call void asm sideeffect inteldialect "cpuid", "~{eax},~{ebx},~{ecx},~{edx},~{dirflag},~{fpsr},~{flags}"()
}

typedef struct {
  int a;
  int b;
} A, *pA;

typedef struct {
  int b1;
  A   b2;
} B;

typedef struct {
  int c1;
  A   c2;
  int c3;
  B   c4;
} C, *pC;

void t39() {
// CHECK-LABEL: define{{.*}} void @t39
  __asm mov eax, [eax].A.b
// CHECK: mov eax, [eax + $$4]
  __asm mov eax, [eax] A.b
// CHECK: mov eax, [eax + $$4]
  __asm mov eax, [eax] pA.b
// CHECK: mov eax, [eax + $$4]
  __asm mov eax, fs:[0] A.b
// CHECK: mov eax, fs:[$$4]
  __asm mov eax, [eax].B.b2.a
// CHECK: mov eax, [eax + $$4]
  __asm mov eax, [eax] B.b2.b
// CHECK: mov eax, [eax + $$8]
  __asm mov eax, fs:[0] C.c2.b
// CHECK: mov eax, fs:[$$8]
  __asm mov eax, [eax]C.c4.b2.b
// CHECK: mov eax, [eax + $$24]
  __asm mov eax, [eax]pC.c4.b2.b
// CHECK: mov eax, [eax + $$24]
// CHECK: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t40(float a) {
// CHECK-LABEL: define{{.*}} void @t40
  int i;
  __asm fld a
// CHECK: fld dword ptr $1
  __asm fistp i
// CHECK: fistp dword ptr $0
// CHECK: "=*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}}, float* elementtype(float) %{{.*}})
}

void t41(unsigned short a) {
// CHECK-LABEL: define{{.*}} void @t41(i16 noundef zeroext %a)
  __asm mov cs, a;
// CHECK: mov cs, $0
  __asm mov ds, a;
// CHECK: mov ds, $1
  __asm mov es, a;
// CHECK: mov es, $2
  __asm mov fs, a;
// CHECK: mov fs, $3
  __asm mov gs, a;
// CHECK: mov gs, $4
  __asm mov ss, a;
// CHECK: mov ss, $5
// CHECK: "*m,*m,*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(i16* {{.*}}, i16* {{.*}}, i16* {{.*}}, i16* {{.*}}, i16* {{.*}}, i16* {{.*}})
}

void t42() {
// CHECK-LABEL: define{{.*}} void @t42(
  int flags;
  __asm mov flags, eax
// CHECK: mov $0, eax
// CHECK: "=*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %flags)
}

void t42b() {
// CHECK-LABEL: define{{.*}} void @t42b(
  int mxcsr;
  __asm mov mxcsr, eax
// CHECK: mov $0, eax
// CHECK: "=*m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %mxcsr)
}

void t43() {
// CHECK-LABEL: define{{.*}} void @t43
  C strct;
// Work around PR20368: These should be single line blocks
 __asm { mov eax, 4[strct.c1] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 4[strct.c3 + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 8[strct.c2.a + 4 + 32*2 - 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$72]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 12[4 + strct.c2.b] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$16]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 4[4 + strct.c4.b2.b + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$12]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 4[64 + strct.c1 + (2*32)] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$132]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, 4[64 + strct.c2.a - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [strct.c4.b1 + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [strct.c4.b2.a + 4 + 32*2 - 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$64]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [4 + strct.c1] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$4]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [4 + strct.c2.b + 4] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$8]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [64 + strct.c3 + (2*32)] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0[$$128]", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
  __asm { mov eax, [64 + strct.c4.b2.b - 2*32] }
// CHECK: call void asm sideeffect inteldialect "mov eax, $0", "*m,~{eax},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %{{.*}})
}

void t44() {
  // CHECK-LABEL: define{{.*}} void @t44
  __asm {
    mov cr0, eax
    mov cr2, ebx
    mov cr3, ecx
    mov cr4, edx
  }
  // CHECK: call void asm sideeffect inteldialect "mov cr0, eax\0A\09mov cr2, ebx\0A\09mov cr3, ecx\0A\09mov cr4, edx", "~{cr0},~{cr2},~{cr3},~{cr4},~{dirflag},~{fpsr},~{flags}"()
}

void t45() {
  // CHECK-LABEL: define{{.*}} void @t45
  __asm {
    mov dr0, eax
    mov dr1, ebx
    mov dr2, ebx
    mov dr3, ecx
    mov dr6, edx
    mov dr7, ecx
  }
  // CHECK: call void asm sideeffect inteldialect "mov dr0, eax\0A\09mov dr1, ebx\0A\09mov dr2, ebx\0A\09mov dr3, ecx\0A\09mov dr6, edx\0A\09mov dr7, ecx", "~{dr0},~{dr1},~{dr2},~{dr3},~{dr6},~{dr7},~{dirflag},~{fpsr},~{flags}"()
}

void t46() {
  // CHECK-LABEL: define{{.*}} void @t46
  __asm add eax, -128[eax]
  // CHECK: call void asm sideeffect inteldialect "add eax, [eax + $$-128]", "~{eax},~{flags},~{dirflag},~{fpsr},~{flags}"()
}

void dot_operator(){
  // CHECK-LABEL: define{{.*}} void @dot_operator
	__asm { mov eax, 3[ebx]A.b}
  // CHECK: call void asm sideeffect inteldialect "mov eax, [ebx + $$7]", "~{eax},~{dirflag},~{fpsr},~{flags}"
}

void call_clobber() {
  __asm call t41
  // CHECK-LABEL: define{{.*}} void @call_clobber
  // CHECK: call void asm sideeffect inteldialect "call dword ptr ${0:P}", "*m,~{dirflag},~{fpsr},~{flags}"(void (i16)* elementtype(void (i16)) @t41)
}

void xgetbv() {
  __asm xgetbv
}
// CHECK-LABEL: define{{.*}} void @xgetbv()
// CHECK: call void asm sideeffect inteldialect "xgetbv", "~{eax},~{edx},~{dirflag},~{fpsr},~{flags}"()

void label1() {
  __asm {
    label:
    jmp label
  }
  // CHECK-LABEL: define{{.*}} void @label1()
  // CHECK: call void asm sideeffect inteldialect "{{.*}}__MSASMLABEL_.${:uid}__label:\0A\09jmp {{.*}}__MSASMLABEL_.${:uid}__label", "~{dirflag},~{fpsr},~{flags}"() [[ATTR1:#[0-9]+]]
}

void label2() {
  __asm {
    jmp label
    label:
  }
  // CHECK-LABEL: define{{.*}} void @label2
  // CHECK: call void asm sideeffect inteldialect "jmp {{.*}}__MSASMLABEL_.${:uid}__label\0A\09{{.*}}__MSASMLABEL_.${:uid}__label:", "~{dirflag},~{fpsr},~{flags}"()
}

void label3() {
  __asm {
    label:
    mov eax, label
  }
  // CHECK-LABEL: define{{.*}} void @label3
  // CHECK: call void asm sideeffect inteldialect "{{.*}}__MSASMLABEL_.${:uid}__label:\0A\09mov eax, {{.*}}__MSASMLABEL_.${:uid}__label", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void label4() {
  __asm {
    label:
    mov eax, [label]
  }
  // CHECK-LABEL: define{{.*}} void @label4
  // CHECK: call void asm sideeffect inteldialect "{{.*}}__MSASMLABEL_.${:uid}__label:\0A\09mov eax, {{.*}}__MSASMLABEL_.${:uid}__label", "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void label5() {
  __asm {
    jmp dollar_label$
    dollar_label$:
  }
  // CHECK-LABEL: define{{.*}} void @label5
  // CHECK: call void asm sideeffect inteldialect "jmp {{.*}}__MSASMLABEL_.${:uid}__dollar_label$$\0A\09{{.*}}__MSASMLABEL_.${:uid}__dollar_label$$:", "~{dirflag},~{fpsr},~{flags}"()
}

void label6(){
  __asm {
      jmp short label
      jc  short label
      jz  short label
    label:
  }
  // CHECK-LABEL: define{{.*}} void @label6
  // CHECK: jmp {{.*}}__MSASMLABEL_.${:uid}__label\0A\09jc {{.*}}__MSASMLABEL_.${:uid}__label\0A\09jz {{.*}}__MSASMLABEL_.${:uid}__label\0A\09{{.*}}__MSASMLABEL_.${:uid}__label:"
}

// Don't include mxcsr in the clobber list.
void mxcsr() {
  char buf[4096];
  __asm fxrstor buf
}
// CHECK-LABEL: define{{.*}} void @mxcsr
// CHECK: call void asm sideeffect inteldialect "fxrstor $0", "=*m,~{fpcr},~{dirflag},~{fpsr},~{flags}"

// Make sure we can find the register for the dirflag for popfd
void dirflag() {
  __asm popfd
}
// CHECK-LABEL: define{{.*}} void @dirflag
// CHECK: call void asm sideeffect inteldialect "popfd", "~{dirflag},~{flags},~{esp},~{dirflag},~{fpsr},~{flags}"

typedef union _LARGE_INTEGER {
  struct {
    unsigned int LowPart;
    unsigned int  HighPart;
  };
  struct {
    unsigned int LowPart;
    unsigned int  HighPart;
  } u;
  unsigned long long QuadPart;
} LARGE_INTEGER, *PLARGE_INTEGER;

int test_indirect_field(LARGE_INTEGER LargeInteger) {
    __asm mov     eax, LargeInteger.LowPart
}
// CHECK-LABEL: define{{.*}} i32 @test_indirect_field(
// CHECK: call i32 asm sideeffect inteldialect "mov eax, $1",

// MS ASM containing labels must not be duplicated (PR23715).
// CHECK: attributes [[ATTR1]] = {
// CHECK-NOT: noduplicate
// CHECK-SAME: }{{$}}
