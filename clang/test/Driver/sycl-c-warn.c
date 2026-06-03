// Verify that a .c file compiled with -fsycl is an error.
// RUN: not %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix ERR %s
// ERR: error: invalid argument '{{.*}}sycl-c-warn.c' not allowed with '-fsycl'

// Verify that explicitly forcing -x c with -fsycl is also an error.
// RUN: not %clang -### -fsycl -x c %s 2>&1 | FileCheck -check-prefix ERR_XC %s
// ERR_XC: error: invalid argument '-x c' not allowed with '-fsycl'
