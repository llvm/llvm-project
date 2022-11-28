// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | opt -passes=strip -S | FileCheck %s

// Some test cases for MS inline asm support from Mozilla code base.

void invoke_copy_to_stack() {}

void invoke(void* that, unsigned methodIndex,
            unsigned paramCount, void* params)
{
// CHECK: @invoke
// CHECK: %5 = alloca ptr, align 4
// CHECK: %6 = alloca i32, align 4
// CHECK: %7 = alloca i32, align 4
// CHECK: %8 = alloca ptr, align 4
// CHECK: store ptr %0, ptr %5, align 4
// CHECK: store i32 %1, ptr %6, align 4
// CHECK: store i32 %2, ptr %7, align 4
// CHECK: store ptr %3, ptr %8, align 4
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov edx,$1
// CHECK-SAME: test edx,edx
// CHECK-SAME: jz {{[^_]*}}__MSASMLABEL_.${:uid}__noparams
//                ^ Can't use {{.*}} here because the matching is greedy.
// CHECK-SAME: mov eax,edx
// CHECK-SAME: shl eax,$$3
// CHECK-SAME: sub esp,eax
// CHECK-SAME: mov ecx,esp
// CHECK-SAME: push $0
// CHECK-SAME: call dword ptr ${2:P}
// CHECK-SAME: {{.*}}__MSASMLABEL_.${:uid}__noparams:
// CHECK-SAME: mov ecx,$3
// CHECK-SAME: push ecx
// CHECK-SAME: mov edx,[ecx]
// CHECK-SAME: mov eax,$4
// CHECK-SAME: call dword ptr[edx + eax * $$4]
// CHECK-SAME: mov esp,ebp
// CHECK-SAME: pop ebp
// CHECK-SAME: ret
// CHECK: "=*m,*m,*m,*m,*m,~{eax},~{ebp},~{ecx},~{edx},~{flags},~{esp},~{dirflag},~{fpsr},~{flags}"
// CHECK: (ptr elementtype(ptr) %8, ptr elementtype(i32) %7, ptr elementtype(void (...)) @invoke_copy_to_stack, ptr elementtype(ptr) %5, ptr elementtype(i32) %6)
// CHECK: ret void
    __asm {
        mov     edx,paramCount
        test    edx,edx
        jz      noparams
        mov     eax,edx
        shl     eax,3
        sub     esp,eax
        mov     ecx,esp
        push    params
        call    invoke_copy_to_stack
noparams:
        mov     ecx,that
        push    ecx
        mov     edx,[ecx]
        mov     eax,methodIndex
        call    dword ptr[edx+eax*4]
        mov     esp,ebp
        pop     ebp
        ret
    }
}

