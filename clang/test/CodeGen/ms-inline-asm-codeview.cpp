// RUN: %clang_cc1 -triple i386-pc-windows-msvc -fasm-blocks -gcodeview \
// RUN:   -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

#line 100 "t.cpp"
int a, b;

int main(int argc, char **argv)
{
    __asm
    {
        lea eax, a
        mov dword ptr [eax], 1

        lea ebx, b
        mov dword ptr [ebx], 1

        mov eax, [eax]
        add [ebx], eax

        inc eax

        imul dword ptr [ebx]
        mov [ebx], eax
    }

    return 0;
}

// CHECK: call i32 asm sideeffect inteldialect
// CHECK-SAME: !srcloc ![[SRCLOC:[0-9]+]]
// CHECK: ![[SRCLOC]] = !{i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i64 {{[0-9]+}}, ![[DBGLOCS:[0-9]+]]}
// CHECK: ![[DBGLOCS]] = !{!"inlineasm.dbg.offset", i32 0, i32 106, i32 9, i32 12, i32 107, i32 9, i32 38, i32 109, i32 9, i32 51, i32 110, i32 9, i32 77, i32 112, i32 9, i32 93, i32 113, i32 9, i32 109, i32 115, i32 9, i32 118, i32 117, i32 9, i32 140, i32 118, i32 9}
