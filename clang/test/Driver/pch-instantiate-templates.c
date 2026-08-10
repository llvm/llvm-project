// CL driver test cases
// RUN: %clang_cl -### /Yc /Fpfoo.pch /Fofoo.obj -- %s 2>&1 | FileCheck --check-prefix=CLANG_CL_YC %s
// RUN: %clang_cl -### /Yc /Fpfoo.pch /Fofoo.obj -fno-pch-instantiate-templates -- %s 2>&1 | FileCheck --check-prefix=CLANG_CL_YC_DISABLE %s

// CLANG_CL_YC: "-fpch-instantiate-templates"
// CLANG_CL_YC_DISABLE-NOT: "-fpch-instantiate-templates"

// GCC driver test cases
// RUN: %clang -### -x c-header %s -o %t/foo.pch 2>&1 | FileCheck -check-prefix=GCC_DEFAULT %s
// RUN: %clang -### -x c-header %s -o %t/foo.pch -fpch-instantiate-templates 2>&1 | FileCheck -check-prefix=GCC_DEFAULT_ENABLE %s

// GCC_DEFAULT-NOT: "-fpch-instantiate-templates"
// GCC_DEFAULT_ENABLE: "-fpch-instantiate-templates"
