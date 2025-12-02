// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=x86-precise
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=x86-sloppy

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=aarch64-precise
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=aarch64-sloppy

// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=riscv-precise
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=riscv-sloppy

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=x86-precise -xc++
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=x86-sloppy -xc++

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=aarch64-precise -xc++
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=aarch64-sloppy -xc++

// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog %s -verify=riscv-precise -xc++
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -O1 -emit-codegen-only -Rpass-analysis=prologepilog -sloppy-temporary-lifetimes %s -verify=riscv-sloppy -xc++


typedef struct { char x[32]; } A;
typedef struct { char *w, *x, *y, *z; } B;

void useA(A);
void useB(B);
A genA(void);
B genB(void);

void t1(int c) {
  // x86-precise-remark@-1 {{40 stack bytes}}
  // x86-sloppy-remark@-2 {{72 stack bytes}}
  // aarch64-precise-remark@-3 {{48 stack bytes}}
  // aarch64-sloppy-remark@-4 {{80 stack bytes}}
  // riscv-precise-remark@-5 {{48 stack bytes}}
  // riscv-sloppy-remark@-6 {{80 stack bytes}}

  if (c)
    useA(genA());
  else
    useA(genA());
}

void t2(void) {
  // x86-precise-remark@-1 {{40 stack bytes}}
  // x86-sloppy-remark@-2 {{72 stack bytes}}
  // aarch64-precise-remark@-3 {{48 stack bytes}}
  // aarch64-sloppy-remark@-4 {{80 stack bytes}}
  // riscv-precise-remark@-5 {{48 stack bytes}}
  // riscv-sloppy-remark@-6 {{80 stack bytes}}

  useA(genA());
  useA(genA());
}

void t3(void) {
  // x86-precise-remark@-1 {{40 stack bytes}}
  // x86-sloppy-remark@-2 {{72 stack bytes}}
  // aarch64-precise-remark@-3 {{48 stack bytes}}
  // aarch64-sloppy-remark@-4 {{80 stack bytes}}
  // riscv-precise-remark@-5 {{48 stack bytes}}
  // riscv-sloppy-remark@-6 {{80 stack bytes}}

  useB(genB());
  useB(genB());
}

#ifdef __cplusplus
struct C {
  char x[24];
  char *ptr;
  ~C() {};
};

void useC(C);
C genC(void);

// This case works in C++, since its AST is structured slightly differently
// than it is in C (CompundStmt/ExprWithCleanup/CallExpr vs CompundStmt/CallExpr).
void t4() {
  // x86-precise-remark@-1 {{40 stack bytes}}
  // x86-sloppy-remark@-2 {{72 stack bytes}}
  // aarch64-precise-remark@-3 {{48 stack bytes}}
  // aarch64-sloppy-remark@-4 {{80 stack bytes}}
  // riscv-precise-remark@-5 {{48 stack bytes}}
  // riscv-sloppy-remark@-6 {{80 stack bytes}}

  useC(genC());
  useC(genC());
}
#endif
