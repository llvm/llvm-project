// RUN: %clang_cc1 -O2 -emit-llvm %s -o - -triple aarch64 | FileCheck %s

int test_cceq(int a, int* b) {
// CHECK-LABEL: @test_cceq
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cceq},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cceq"(*b));
  return a;
}

int test_ccne(int a, int* b) {
// CHECK-LABEL: @test_ccne
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccne},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccne"(*b));
  return a;
}

int test_cccs(int a, int* b) {
// CHECK-LABEL: @test_cccs
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cccs},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cccs"(*b));
  return a;
}

int test_cchs(int a, int* b) {
// CHECK-LABEL: @test_cchs
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cchs},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cchs"(*b));
  return a;
}

int test_cccc(int a, int* b) {
// CHECK-LABEL: @test_cccc
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cccc},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cccc"(*b));
  return a;
}

int test_cclo(int a, int* b) {
// CHECK-LABEL: @test_cclo
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cclo},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cclo"(*b));
  return a;
}

int test_ccmi(int a, int* b) {
// CHECK-LABEL: @test_ccmi
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccmi},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccmi"(*b));
  return a;
}

int test_ccpl(int a, int* b) {
// CHECK-LABEL: @test_ccpl
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccpl},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccpl"(*b));
  return a;
}

int test_ccvs(int a, int* b) {
// CHECK-LABEL: @test_ccvs
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccvs},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccvs"(*b));
  return a;
}

int test_ccvc(int a, int* b) {
// CHECK-LABEL: @test_ccvc
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccvc},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccvc"(*b));
  return a;
}

int test_cchi(int a, int* b) {
// CHECK-LABEL: @test_cchi
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cchi},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cchi"(*b));
  return a;
}

int test_ccls(int a, int* b) {
// CHECK-LABEL: @test_ccls
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccls},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccls"(*b));
  return a;
}


int test_ccge(int a, int* b) {
// CHECK-LABEL: @test_ccge
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccge},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccge"(*b));
  return a;
}

int test_cclt(int a, int* b) {
// CHECK-LABEL: @test_cclt
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@cclt},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@cclt"(*b));
  return a;
}

int test_ccgt(int a, int* b) {
// CHECK-LABEL: @test_ccgt
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccgt},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccgt"(*b));
  return a;
}

int test_ccle(int a, int* b) {
// CHECK-LABEL: @test_ccle
// CHECK: = tail call { i32, i32 } asm "ands ${0:w}, ${0:w}, #3", "=r,={@ccle},0"(i32 %a)
  asm("ands %w[a], %w[a], #3"
      : [a] "+r"(a), "=@ccle"(*b));
  return a;
}
