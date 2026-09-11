// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -x c -emit-pch -o %t/a.c.ast %t/a.c
// RUN: %clang_cc1 -x c -emit-pch -o %t/b.c.ast %t/b.c
// RUN: %clang_cc1 -x c -emit-pch -o %t/c.c.ast %t/c.c

// RUN: %clang_extdef_map %t/a.c %t/b.c %t/c.c -- -c -x c > %t/externalDefMap.tmp1.txt
// RUN: sed -e 's|\.c$|.c.ast|g' %t/externalDefMap.tmp1.txt > %t/externalDefMap.tmp2.txt
// RUN: sed -e 's|%t\/||g' %t/externalDefMap.tmp2.txt > %t/externalDefMap.txt

// RUN: %clang_cc1 -analyze -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c 2>&1 | FileCheck %s

//--- main.c

// expected-no-diagnostics
// CHECK: CTU loaded AST file: b.c.ast

int fn(void);

int main(int argc, char* argv[]) {
  return fn();
}

//--- a.c

int fn(void) __attribute__((weak));

int fn(void) {
   return 0;
}

//--- b.c

int fn(void) {
   return 1;
}

//--- c.c

int fn(void) __attribute__((weak));

int fn(void) {
   return 0;
}
