// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -x c -emit-pch -o %t/import.c.ast %t/import.c

// RUN: %clang_extdef_map %t/import.c -- -c -x c > %t/externalDefMap.tmp.txt
// RUN: sed 's/$/.ast/' %t/externalDefMap.tmp.txt > %t/externalDefMap.txt

// RUN: %clang_cc1 -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c 2>&1 | FileCheck %s

//--- main.c

// expected-no-diagnostics
// CHECK: CTU loaded AST file:

typedef struct X_s X_t;

long f_import(struct X_s *xPtr);

static void f_main(struct X_s *xPtr) {
  f_import(xPtr);
}

//--- import.c

typedef struct Y_s Y_t;

struct Y_s {
};

struct X_s {
  Y_t y;
};

long f_import(struct X_s *xPtr) {
  if (xPtr != 0) {
  }
  return 0;
}
