// Verify that a .c file compiled with -fsycl is an error.
// RUN: not %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix ERR %s
// ERR: error: invalid argument '{{.*}}sycl-c-warn.c' not allowed with '-fsycl'
