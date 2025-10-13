// RUN: %clang -fbounds-safety -### %s 2>&1 | FileCheck --check-prefix=DEFAULT %s

// RUN: %clang -fbounds-safety -fbounds-safety-soft-traps=disabled -### %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: %clang -fbounds-safety -fbounds-safety-soft-traps=call-with-str -### %s 2>&1 | FileCheck --check-prefix=CWS %s
// RUN: %clang -fbounds-safety -fbounds-safety-soft-traps=call-with-code -### %s 2>&1 | FileCheck --check-prefix=CWC %s

// DEFAULT-NOT: -fbounds-safety-soft-traps=
// DISABLED:  -fbounds-safety-soft-traps=disabled
// CWS:  -fbounds-safety-soft-traps=call-with-str
// CWC:  -fbounds-safety-soft-traps=call-with-code
