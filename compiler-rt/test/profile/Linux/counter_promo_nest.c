// RUN: rm -fr %t.promo.prof
// RUN: rm -fr %t.nopromo.prof
// RUN: %clang_pgogen=%t.promo.prof/ -o %t.promo.gen -O2 %s
// RUN: %clang_pgogen=%t.promo.prof/ -o %t.promo.gen.ll -emit-llvm -S -O2 %s
// RUN: cat %t.promo.gen.ll | FileCheck --check-prefix=PROMO %s
// RUN: %run %t.promo.gen
// RUN: llvm-profdata merge -o %t.promo.profdata %t.promo.prof/
// RUN: llvm-profdata show --counts --all-functions %t.promo.profdata  > %t.promo.dump
// RUN: %clang_pgogen=%t.nopromo.prof/ -mllvm -do-counter-promotion=false -o %t.nopromo.gen -O2 %s
// RUN: %run %t.nopromo.gen
// RUN: llvm-profdata merge -o %t.nopromo.profdata %t.nopromo.prof/
// RUN: llvm-profdata show --counts --all-functions %t.nopromo.profdata  > %t.nopromo.dump
// RUN: diff %t.promo.profdata %t.nopromo.profdata
int g;
__attribute__((noinline)) void bar() {
 g++;
}

extern int printf(const char*,...);

int c = 10;

int main()
// PROMO-LABEL: @main
// PROMO: load{{.*}}@__profc_main{{.*}}
// PROMO-NEXT: add
// PROMO-NEXT: store{{.*}}@__profc_main{{.*}}
// PROMO: load{{.*}}@__profc_main{{.*}}
// PROMO-NEXT: add
// PROMO-NEXT: store{{.*}}@__profc_main{{.*}}
// PROMO-NEXT: load{{.*}}@__profc_main{{.*}}
// PROMO-NEXT: add
// PROMO-NEXT: store{{.*}}@__profc_main{{.*}}
{
  int i, j, k;

  g = 0;
  for (i = 0; i < c; i++)
    for (j = 0; j < c; j++)
       for (k = 0; k < c; k++)
           bar();

  for (i = 0; i < c; i++)
    for (j = 0; j < 10*c;j++)
        bar();

  for (i = 0; i < 100*c; i++)
    bar();

  return 0;
}
