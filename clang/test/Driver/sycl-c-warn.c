// Verify that a .c file compiled with -fsycl is treated as C++ with a warning.
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix WARN %s
// WARN: warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated [-Wdeprecated]

// Verify that explicitly forcing -x c with -fsycl is an error.
// RUN: not %clang -### -fsycl -x c %s 2>&1 | FileCheck -check-prefix ERR %s
// ERR: error: invalid argument '-x c' not allowed with '-fsycl'
