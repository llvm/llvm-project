// Verify that explicitly forcing -x c with -fsycl is an error.
// RUN: not %clang -### -fsycl -x c %s 2>&1 | FileCheck -check-prefix ERR_XC %s
// ERR_XC: error: invalid argument '-x c' not allowed with '-fsycl'

// Verify that clang-cl /TC (forces all inputs to C) with -fsycl is an error.
// RUN: not %clang_cl -### -fsycl /TC %s 2>&1 | FileCheck -check-prefix ERR_TC %s
// ERR_TC: error: invalid argument '/TC' not allowed with '-fsycl'

// Verify that clang-cl /Tc (forces a specific file to C) with -fsycl is an error.
// RUN: not %clang_cl -### -fsycl /Tc%s 2>&1 | FileCheck -check-prefix ERR_Tc %s
// ERR_Tc: error: invalid argument '/Tc{{.*}}sycl-force-c-input-error.cpp' not allowed with '-fsycl'
