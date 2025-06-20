// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -msingle-float 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -msoft-float 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -mfpu=32 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -mfpu=0 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -mfpu=none 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlasx -mno-lsx 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LASX_FPU128 %s

// ERROR_LASX_FPU64: error: wrong fpu width; LASX depends on 64-bit FPU
// ERROR_LASX_FPU128: error: invalid option combination; LASX depends on LSX
