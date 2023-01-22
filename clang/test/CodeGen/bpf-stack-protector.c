// REQUIRES: bpf-registered-target

// RUN %clang -target bpf -S -emit-llvm -o - %s -fno-stack-protector 2>&1 \
// RUN        | FileCheck -check-prefix=OFF -check-prefix=COMMON %s

// RUN: %clang -target bpf -S -emit-llvm -o - %s -fstack-protector 2>&1 \
// RUN:        | FileCheck -check-prefix=ON -check-prefix=COMMON %s

// RUN: %clang -target bpf -S -emit-llvm -o - %s -fstack-protector-all 2>&1 \
// RUN:        | FileCheck -check-prefix=ALL -check-prefix=COMMON %s

// RUN: %clang -target bpf -S -emit-llvm -o - %s -fstack-protector-strong 2>&1 \
// RUN:        | FileCheck -check-prefix=STRONG -check-prefix=COMMON %s

typedef __SIZE_TYPE__ size_t;

int printf(const char * _Format, ...);
size_t strlen(const char *s);
char *strcpy(char *s1, const char *s2);

//     OFF-NOT: warning
//          ON: warning: ignoring '-fstack-protector'
//         ALL: warning: ignoring '-fstack-protector-all'
//      STRONG: warning: ignoring '-fstack-protector-strong'
// COMMON-SAME: option as it is not currently supported for target 'bpf'

// COMMON: define {{.*}}void @test1(ptr noundef %msg) #[[A:.*]] {
void test1(const char *msg) {
  char a[strlen(msg) + 1];
  strcpy(a, msg);
  printf("%s\n", a);
}

// COMMON-NOT: attributes #[[A]] = {{.*}} ssp
