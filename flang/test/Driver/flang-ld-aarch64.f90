! Check linker flags for AArch64 linux, since it needs both libgcc and 
! compiler-rt, with compiler-rt second when -rtlib=libgcc.

! RUN: %flang -### -rtlib=libgcc --target=aarch64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s

! CHECK-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! CHECK-SAME: "-lflang_rt.runtime" "-lm"
! CHECK-SAME: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
! CHECK-SAME: "--as-needed" "{{.*}}{{\\|/}}libclang_rt.builtins.a" "--no-as-needed"
