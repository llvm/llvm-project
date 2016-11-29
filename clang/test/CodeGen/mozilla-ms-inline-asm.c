// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | opt -strip -S | FileCheck %s

// Some test cases for MS inline asm support from Mozilla code base.

void invoke_copy_to_stack() {}

void invoke(void* that, unsigned methodIndex,
            unsigned paramCount, void* params)
{
// CHECK: @invoke
// CHECK: %5 = alloca i8*, align 4
// CHECK: %6 = alloca i32, align 4
// CHECK: %7 = alloca i32, align 4
// CHECK: %8 = alloca i8*, align 4
// CHECK: store i8* %0, i8** %5, align 4
// CHECK: store i32 %1, i32* %6, align 4
// CHECK: store i32 %2, i32* %7, align 4
// CHECK: store i8* %3, i8** %8, align 4
// CHECK: call void asm sideeffect inteldialect
// CHECK: mov edx,dword ptr $1
// CHECK: test edx,edx
// CHECK: jz {{[^_]*}}__MSASMLABEL_.${:uid}__noparams
//             ^ Can't use {{.*}} here because the matching is greedy.
// CHECK: mov eax,edx
// CHECK: shl eax,$$3
// CHECK: sub esp,eax
// CHECK: mov ecx,esp
// CHECK: push dword ptr $0
// CHECK: call dword ptr $2
// CHECK: {{.*}}__MSASMLABEL_.${:uid}__noparams:
// CHECK: mov ecx,dword ptr $3
// CHECK: push ecx
// CHECK: mov edx,[ecx]
// CHECK: mov eax,dword ptr $4
// CHECK: call dword ptr[edx+eax*$$4]
// CHECK: mov esp,ebp
// CHECK: pop ebp
// CHECK: ret
// CHECK: "=*m,*m,*m,*m,*m,~{eax},~{ebp},~{ecx},~{edx},~{flags},~{esp},~{dirflag},~{fpsr},~{flags}"
// CHECK: (i8** %8, i32* %7, void (...)* bitcast (void ()* @invoke_copy_to_stack to void (...)*), i8** %5, i32* %6)
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

