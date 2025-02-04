!RUN: %flang --target=aarch64-unknown-linux-gnu -fuse-ld=ld -fopenmp -rtlib=libgcc -### %s 2>&1 | FileCheck --check-prefixes=GCC %s
!RUN: %flang --target=aarch64-unknown-linux-gnu -fuse-ld=ld -fopenmp -rtlib=compiler-rt -### %s 2>&1 | FileCheck --check-prefixes=CRT %s

!GCC: -latomic
!CRT-NOT: -latomic
