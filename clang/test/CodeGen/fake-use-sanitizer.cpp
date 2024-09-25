// RUN: %clang_cc1 %s -O1 -emit-llvm -fextend-lifetimes -fsanitize=null -fsanitize-trap=null -o - | FileCheck --check-prefix TRAP %s
// RUN: %clang_cc1 %s -O1 -emit-llvm -fextend-lifetimes -o - | FileCheck --check-prefix FAKEUSE %s

// With -fextend-lifetimes the compiler generated a fake.use of a
// reference variable at the end of the function, in the cleanup block. This prompted the 
// address sanitizer to verify that the variable has been allocated properly, even when 
// the function returns early.
// We check that sanitizers are disabled for code generated for the benefit of fake.use 
// intrinsics, as well as that the fake.use is no longer generated in the cleanup block.
// It should be generated in the block preceding the cleanup block instead.

struct A { short s1, s2; };
extern A& getA();

void foo()
{
  auto& va = getA();
  short s = va.s1 & ~va.s2;
  if (s == 0)
    return;

  auto& vb = getA();
}

// TRAP:         define{{.*}}foo
// TRAP:         [[COMPARE:%[^\s]*]] = icmp eq
// TRAP-NOT:     br i1 [[COMPARE]]{{.*}}trap
// TRAP:         br i1 [[COMPARE]]{{.*}}%if.end
// TRAP-NOT:     trap:
// TRAP:         if.end:
// TRAP-NOT:     call{{.*}}llvm.trap

// FAKEUSE:      if.end:
// FAKEUSE-NEXT: [[CALL:%[^\s]*]] = {{.*}}call{{.*}}getA
// FAKEUSE-NOT:  br{{.*}}cleanup
// FAKEUSE:      call{{.*}}fake.use({{.*}}[[CALL]])
// FAKEUSE:      cleanup:
