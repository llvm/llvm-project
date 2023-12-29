// RUN: %clang_cc1 -triple arm64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

// The only part clang really deals with is the lvalue/rvalue
// distinction on constraints. It's sufficient to emit llvm and make
// sure that's sane.

long var;

void test_generic_constraints(int var32, long var64) {
    asm("add %0, %1, %1" : "=r"(var32) : "0"(var32));
// CHECK: [[R32_ARG:%[a-zA-Z0-9]+]] = load i32, ptr
// CHECK: call i32 asm "add $0, $1, $1", "=r,0"(i32 [[R32_ARG]])

    asm("add %0, %1, %1" : "=r"(var64) : "0"(var64));
// CHECK: [[R32_ARG:%[a-zA-Z0-9]+]] = load i64, ptr
// CHECK: call i64 asm "add $0, $1, $1", "=r,0"(i64 [[R32_ARG]])

    asm("ldr %0, %1" : "=r"(var32) : "m"(var));
    asm("ldr %0, [%1]" : "=r"(var64) : "r"(&var));
// CHECK: call i32 asm "ldr $0, $1", "=r,*m"(ptr elementtype(i64) @var)
// CHECK: call i64 asm "ldr $0, [$1]", "=r,r"(ptr @var)
}

float f;
double d;
void test_constraint_w(void) {
    asm("fadd %s0, %s1, %s1" : "=w"(f) : "w"(f));
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load float, ptr @f
// CHECK: call float asm "fadd ${0:s}, ${1:s}, ${1:s}", "=w,w"(float [[FLT_ARG]])

    asm("fadd %d0, %d1, %d1" : "=w"(d) : "w"(d));
// CHECK: [[DBL_ARG:%[a-zA-Z_0-9]+]] = load double, ptr @d
// CHECK: call double asm "fadd ${0:d}, ${1:d}, ${1:d}", "=w,w"(double [[DBL_ARG]])
}

void test_constraints_immed(void) {
    asm("add x0, x0, %0" : : "I"(4095) : "x0");
    asm("and w0, w0, %0" : : "K"(0xaaaaaaaa) : "w0");
    asm("and x0, x0, %0" : : "L"(0xaaaaaaaaaaaaaaaa) : "x0");
// CHECK: call void asm sideeffect "add x0, x0, $0", "I,~{x0}"(i32 4095)
// CHECK: call void asm sideeffect "and w0, w0, $0", "K,~{w0}"(i32 -1431655766)
// CHECK: call void asm sideeffect "and x0, x0, $0", "L,~{x0}"(i64 -6148914691236517206)
}

void test_constraint_S(void) {
    int *addr;
    asm("adrp %0, %1\n\t"
        "add %0, %0, :lo12:%1" : "=r"(addr) : "S"(&var));
// CHECK: call ptr asm "adrp $0, $1\0A\09add $0, $0, :lo12:$1", "=r,S"(ptr @var)
}

void test_constraint_Q(void) {
    int val;
    asm("ldxr %0, %1" : "=r"(val) : "Q"(var));
// CHECK: call i32 asm "ldxr $0, $1", "=r,*Q"(ptr elementtype(i64) @var)
}

void test_gcc_registers(void) {
    register unsigned long reg0 asm("r0") = 0;
    register unsigned long reg1 asm("r1") = 1;
    register unsigned int  reg29 asm("r29") = 2;
    register unsigned int  reg30 asm("r30") = 3;

    // Test remapping register names in register ... asm("rN") statments.
    // rN register operands in these two inline assembly lines
    // should get renamed to valid AArch64 registers.
    asm volatile("hvc #0" : : "r" (reg0), "r" (reg1));
    // CHECK: call void asm sideeffect "hvc #0", "{x0},{x1}"
    asm volatile("hvc #0" : : "r" (reg29), "r" (reg30));
    // CHECK: call void asm sideeffect "hvc #0", "{fp},{lr}"

    // rN registers when used without register ... asm("rN") syntax
    // should not be remapped.
    asm volatile("mov r0, r1\n");
    // CHECK: call void asm sideeffect "mov r0, r1\0A", ""()
}

void test_tied_earlyclobber(void) {
  register int a asm("x1");
  asm("" : "+&r"(a));
  // CHECK: call i32 asm "", "=&{x1},0"(i32 %0)
}

void test_reduced_gpr_constraints(int var32, long var64) {
  asm("add w0, w0, %0" : : "Uci"(var32) : "w0");
// CHECK: [[ARG1:%.+]] = load i32, ptr
// CHECK: call void asm sideeffect "add w0, w0, $0", "@3Uci,~{w0}"(i32 [[ARG1]])
  asm("add x0, x0, %0" : : "Uci"(var64) : "x0");
// CHECK: [[ARG1:%.+]] = load i64, ptr
// CHECK: call void asm sideeffect "add x0, x0, $0", "@3Uci,~{x0}"(i64 [[ARG1]])
  asm("add w0, w0, %0" : : "Ucj"(var32) : "w0");
// CHECK: [[ARG2:%.+]] = load i32, ptr
// CHECK: call void asm sideeffect "add w0, w0, $0", "@3Ucj,~{w0}"(i32 [[ARG2]])
  asm("add x0, x0, %0" : : "Ucj"(var64) : "x0");
// CHECK: [[ARG2:%.+]] = load i64, ptr
// CHECK: call void asm sideeffect "add x0, x0, $0", "@3Ucj,~{x0}"(i64 [[ARG2]])
}
