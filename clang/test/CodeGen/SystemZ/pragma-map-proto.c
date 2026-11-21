// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -o - | FileCheck %s
// REQUIRES: systemz-registered-target

int printf(const char *, ...);
void perror(const char *);
#pragma map(printf, "PRINTF")
#pragma map(perror, "PERROR")

int main() {
  printf("hello");
  perror("hello");
}

// CHECK: call {{.*}} @PRINTF({{.*}} @.str)
// CHECK: call {{.*}} @PERROR({{.*}} @.str)

// CHECK: declare {{.*}} @PRINTF(ptr noundef, ...)
// CHECK: declare {{.*}} @PERROR(ptr noundef)
