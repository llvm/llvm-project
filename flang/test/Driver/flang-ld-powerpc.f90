!! Testing ld command with flang on POWERPC.
!! TODO: The AIX test case is meant to test the behavior of linking the static
!!       libflang_rt.runtime.a, which will be enabled by a new compiler option
!!       -static-libflang_rt in the future. Need to add that option here.

!! Because flang-rt currently only supports
!! LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON, use 
!! resource_dir_with_per_target_subdir as inputs.

! Check powerpc64-ibm-aix 64-bit linking to static flang-rt
! RUN: %flang %s -### 2>&1 \
! RUN:        --target=powerpc64-ibm-aix \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefix=AIX64-LD-PER-TARGET

! AIX64-LD-PER-TARGET-NOT: warning:
! AIX64-LD-PER-TARGET:     "-fc1" "-triple" "powerpc64-ibm-aix"
! AIX64-LD-PER-TARGET-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! AIX64-LD-PER-TARGET:     "{{.*}}ld{{(.exe)?}}"
! AIX64-LD-PER-TARGET-NOT: "-bnso"
! AIX64-LD-PER-TARGET-SAME:     "-b64"
! AIX64-LD-PER-TARGET-SAME:     "-bpT:0x100000000" "-bpD:0x110000000"
! AIX64-LD-PER-TARGET-SAME:     "-lc"
! AIX64-LD-PER-TARGET-SAME:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}powerpc64-ibm-aix{{/|\\\\}}libflang_rt.runtime.a"
! AIX64-LD-PER-TARGET-SAME:     "-lm"
! AIX64-LD-PER-TARGET-SAME:     "-lpthread"

! Check powerpc64le-unknown-linux-gnu 64-bit linking to static flang-rt
! RUN: %flang %s -### 2>&1 \
! RUN:        --target=powerpc64le-unknown-linux-gnu \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefixes=LOP64-LD-PER-TARGET

! LOP64-LD-PER-TARGET-NOT: warning:
! LOP64-LD-PER-TARGET:     "-fc1" "-triple" "powerpc64le-unknown-linux-gnu"
! LOP64-LD-PER-TARGET-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! LOP64-LD-PER-TARGET:     "{{.*}}ld{{(.exe)?}}"
! LOP64-LD-PER-TARGET-NOT: "-bnso"
! LOP64-LD-PER-TARGET-SAME:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}powerpc64le-unknown-linux-gnu{{/|\\\\}}libflang_rt.runtime.a"
! LOP64-LD-PER-TARGET-SAME:     "-lm"
! LOP64-LD-PER-TARGET-SAME:     "-lc"
