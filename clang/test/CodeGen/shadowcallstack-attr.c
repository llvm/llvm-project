// RUN: %clang_cc1 -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=NOTIGNORELISTED %s

// RUN: %clang_cc1 -D ATTR -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

// RUN: echo -e "[shadow-call-stack]\nfun:foo" > %t
// RUN: %clang_cc1 -fsanitize-ignorelist=%t -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

// RUN: %clang_cc1 -triple riscv32-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=NOTIGNORELISTED %s

// RUN: %clang_cc1 -D ATTR -triple riscv32-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

// RUN: echo -e "[shadow-call-stack]\nfun:foo" > %t
// RUN: %clang_cc1 -fsanitize-ignorelist=%t -triple riscv32-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

// RUN: %clang_cc1 -triple riscv64-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=NOTIGNORELISTED %s

// RUN: %clang_cc1 -D ATTR -triple riscv64-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

// RUN: echo -e "[shadow-call-stack]\nfun:foo" > %t
// RUN: %clang_cc1 -fsanitize-ignorelist=%t -triple riscv64-linux-gnu -emit-llvm -o - %s -fsanitize=shadow-call-stack | FileCheck -check-prefix=IGNORELISTED %s

#ifdef ATTR
__attribute__((no_sanitize("shadow-call-stack")))
#endif
int foo(int *a) { return *a; }

// CHECK: define i32 @foo(i32* %a)

// IGNORELISTED-NOT: attributes {{.*}}shadowcallstack{{.*}}
// NOTIGNORELISTED: attributes {{.*}}shadowcallstack{{.*}}
