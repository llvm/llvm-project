// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlsx -msingle-float 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LSX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlsx -msoft-float 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LSX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlsx -mfpu=32 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LSX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlsx -mfpu=0 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LSX_FPU64 %s
// RUN: not %clang --target=loongarch64 %s -fsyntax-only -mlsx -mfpu=none 2>&1 \
// RUN:   FileCheck --check-prefix=ERROR_LSX_FPU64 %s

// ERROR_LSX_FPU64: error: wrong fpu width; LSX depends on 64-bit FPU
