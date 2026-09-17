// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -x c -emit-pch -o %t/a.c.ast %t/a.c
// RUN: %clang_cc1 -x c -emit-pch -o %t/b.c.ast %t/b.c

// RUN: %clang_extdef_map %t/a.c.ast %t/b.c.ast > %t/externalDefMap.tmp.txt 2> %t/extdef_err.txt
// RUN: sed -e 's|%t\/||g' %t/externalDefMap.tmp.txt > %t/externalDefMap.txt
// RUN: sed -e 's|%t\/||g' %t/extdef_err.txt | FileCheck --allow-empty %t/extdef_check

// RUN: %clang_cc1 -analyze -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c 2>&1 | FileCheck %t/main_check

//--- extdef_check

// CHECK-NOT: warning

//--- main_check

// CHECK: CTU loaded AST file: b.c.ast

//--- main.c

// expected-no-diagnostics

int fn(void);

int main(int argc, char* argv[]) {
  return fn();
}

//--- a.c

int fn(void) __attribute__((weak));

int fn(void) {
   return 1;
}

//--- b.c

int fn(void) {
   return 0;
}
