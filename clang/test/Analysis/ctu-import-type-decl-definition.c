// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-pch -o %t/import.ast %t/import.c

// RUN: %clang_cc1 -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c

//--- main.c

// expected-no-diagnostics

typedef struct X_s X_t;
unsigned long f_import(struct X_s *xPtr);

static void freeWriteFileResources(struct X_s *xPtr) {
  f_import(xPtr);
}

//--- import.c

typedef struct Y_s Y_t;

struct Y_s {
};

struct X_s {
  Y_t y;
};

unsigned long f_import(struct X_s *xPtr) {
  if (xPtr != 0) {
  }
  return 0;
}

//--- externalDefMap.txt
13:c:@F@f_import import.ast
