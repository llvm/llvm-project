// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 0 | FileCheck -check-prefix=DEF -check-prefix=NOSSP %s
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 1 | FileCheck -check-prefix=DEF -check-prefix=SSP %s
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 2 | FileCheck -check-prefix=DEF -check-prefix=SSPSTRONG %s
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 3 | FileCheck -check-prefix=DEF -check-prefix=SSPREQ %s

// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=safe-stack | FileCheck -check-prefix=DEF -check-prefix=SAFESTACK-NOSSP %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=safe-stack -stack-protector 0 | FileCheck -check-prefix=DEF -check-prefix=SAFESTACK-NOSSP %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=safe-stack -stack-protector 1 | FileCheck -check-prefix=DEF -check-prefix=SAFESTACK-SSP %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=safe-stack -stack-protector 2 | FileCheck -check-prefix=DEF -check-prefix=SAFESTACK-SSPSTRONG %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=safe-stack -stack-protector 3 | FileCheck -check-prefix=DEF -check-prefix=SAFESTACK-SSPREQ %s

typedef __SIZE_TYPE__ size_t;

int printf(const char * _Format, ...);
size_t strlen(const char *s);
char *strcpy(char *s1, const char *s2);

// DEF: define {{.*}}void @test1(i8* %msg) #[[A:.*]] {
void test1(const char *msg) {
  char a[strlen(msg) + 1];
  strcpy(a, msg);
  printf("%s\n", a);
}

// NOSSP-NOT: attributes #[[A]] = {{.*}} ssp
// SSP: attributes #[[A]] = {{.*}} ssp{{ }}
// SSPSTRONG: attributes #[[A]] = {{.*}} sspstrong
// SSPREQ: attributes #[[A]] = {{.*}} sspreq

// SAFESTACK-NOSSP: attributes #[[A]] = {{.*}} safestack
// SAFESTACK-NOSSP-NOT: ssp

// SAFESTACK-SSP: attributes #[[A]] = {{.*}} safestack ssp{{ }}
// SAFESTACK-SSPSTRONG: attributes #[[A]] = {{.*}} safestack sspstrong
// SAFESTACK-SSPREQ: attributes #[[A]] = {{.*}} safestack sspreq
