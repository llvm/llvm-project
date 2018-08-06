// RUN: %clang_cc1 -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=UNBLACKLISTED %s

// RUN: %clang_cc1 -D ATTR -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=BLACKLISTED %s

// RUN: echo -e "[shadow-call-stack]\nfun:foo" > %t
// RUN: %clang_cc1 -fsanitize-blacklist=%t -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=BLACKLISTED %s

#ifdef ATTR
__attribute__((no_sanitize("shadow-call-stack")))
#endif
int foo(int *a) { return *a; }

// CHECK: define i32 @foo(i32* %a)

// BLACKLISTED-NOT: attributes {{.*}}shadowcallstack{{.*}}
// UNBLACKLISTED: attributes {{.*}}shadowcallstack{{.*}}
