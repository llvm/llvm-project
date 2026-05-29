// Verify that a .c file compiled with -fsycl is treated as C++ with a warning.
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix WARN %s
// WARN: warning: treating 'c' input as 'c++' when -fsycl is used [-Winvalid-command-line-argument]

// Verify that explicitly forcing -x c with -fsycl is an error.
// RUN: not %clang -### -fsycl -x c %s 2>&1 | FileCheck -check-prefix ERR %s
// ERR: error: '-x c' must not be used in conjunction with '-fsycl', which expects C++ source
