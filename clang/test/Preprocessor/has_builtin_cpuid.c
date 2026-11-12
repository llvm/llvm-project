// RUN: %clang_cc1 -fsyntax-only -triple arm64-- -DARM -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-- -DX86 -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple powerpc64-unknown-linux-gnu -DPPC \
// RUN:   -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple riscv32-unknown-linux-gnu -DRISCV \
// RUN:   -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple riscv64-unknown-linux-gnu -DRISCV \
// RUN:   -verify %s
// expected-no-diagnostics
#if __has_builtin(__builtin_cpu_is)
# if defined(ARM)
#   error "ARM shouldn't have __builtin_cpu_is"
# endif
#endif

#if __has_builtin(__builtin_cpu_init)
# if defined(ARM) || defined(PPC)
#   error "ARM/PPC shouldn't have __builtin_cpu_init"
# endif
#else
# ifdef RISCV
#   error "RISCV should have __builtin_cpu_init"
# endif
#endif

#if !__has_builtin(__builtin_cpu_supports)
# if defined(ARM) || defined(X86) || defined(RISCV)
#   error "ARM/X86/RISCV should have __builtin_cpu_supports"
# endif
#endif
