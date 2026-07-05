// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR 4349

union reg
{
    unsigned char b[2][2];
    unsigned short w[2];
    unsigned int d;
};
struct cpu
{
    union reg pc;
};
extern struct cpu cpu;
struct svar
{
    void *ptr;
};
// CHECK: @svars1 = {{(dso_local )?}}global [1 x %struct.svar] [%struct.svar { ptr @cpu }]
struct svar svars1[] =
{
    { &((cpu.pc).w[0]) }
};
// CHECK: @svars2 = {{(dso_local )?}}global [1 x %struct.svar] [%struct.svar { ptr getelementptr (i8, ptr @cpu, i64 1) }]
struct svar svars2[] =
{
    { &((cpu.pc).b[0][1]) }
};
// CHECK: @svars3 = {{(dso_local )?}}global [1 x %struct.svar] [%struct.svar { ptr getelementptr (i8, ptr @cpu, i64 2) }]
struct svar svars3[] =
{
    { &((cpu.pc).w[1]) }
};
// CHECK: @svars4 = {{(dso_local )?}}global [1 x %struct.svar] [%struct.svar { ptr getelementptr (i8, ptr @cpu, i64 3) }]
struct svar svars4[] =
{
    { &((cpu.pc).b[1][1]) }
};
