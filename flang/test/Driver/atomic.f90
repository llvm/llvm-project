!RUN: %flang -fopenmp -rtlib=libgcc -### %s 2>&1 | FileCheck --check-prefixes=GCC %s
!RUN: %flang -fopenmp -rtlib=compiler-rt -### %s 2>&1 | FileCheck --check-prefixes=CRT %s

!GCC: -latomic
!CRT-NOT: -latomic
