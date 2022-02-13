// RUN: %clang_cc1 -triple amdgcn---amdgizcl -debug-info-kind=limited -gno-column-info -O0 -emit-llvm -o - %s | FileCheck %s

typedef struct
{
    int a;
} Struct;

Struct func1(void);

void func2(Struct S);

void func3(void)
{
    // CHECK: call i32 @func1() #{{[0-9]+}}, !dbg ![[LOC:[0-9]+]]
    // CHECK: call void @func2(i32 %{{[0-9]+}}) #{{[0-9]+}}, !dbg ![[LOC]]
    func2(func1());
}

