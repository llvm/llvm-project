!! Testing ld command with flang on POWERPC.
!! TODO: The AIX test case is meant to test the behavior of linking the static
!!       libflang_rt.runtime.a, which will be enabled by a new compiler option
!!       -static-libflang_rt in the future. Need to add that option here.

!! Because flang-rt currently only supports
!! LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON, use 
!! resource_dir_with_per_target_subdir as inputs.

! Check powerpc64-ibm-aix 64-bit linking to static flang-rt by default
! RUN: %flang -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64-ibm-aix \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefix=AIX64-LD-PER-TARGET-DEFAULT

! AIX64-LD-PER-TARGET-DEFAULT:     "-fc1" "-triple" "powerpc64-ibm-aix"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! AIX64-LD-PER-TARGET-DEFAULT:     "{{.*}}ld{{(.exe)?}}"
! AIX64-LD-PER-TARGET-DEFAULT-NOT: "-bnso"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-b64"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-bpT:0x100000000" "-bpD:0x110000000"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-lc"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}powerpc64-ibm-aix{{/|\\\\}}libflang_rt.runtime.a"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-lm"
! AIX64-LD-PER-TARGET-DEFAULT-SAME:     "-lpthread"


! Check powerpc64-ibm-aix 64-bit linking to static flang-rt by option 
! RUN: %flang -static-libflangrt -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64-ibm-aix \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefix=AIX64-LD-PER-TARGET-STATIC

! AIX64-LD-PER-TARGET-STATIC:     "-fc1" "-triple" "powerpc64-ibm-aix"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! AIX64-LD-PER-TARGET-STATIC:     "{{.*}}ld{{(.exe)?}}"
! AIX64-LD-PER-TARGET-STATIC-NOT: "-bnso"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-b64"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-bpT:0x100000000" "-bpD:0x110000000"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-lc"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}powerpc64-ibm-aix{{/|\\\\}}libflang_rt.runtime.a"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-lm"
! AIX64-LD-PER-TARGET-STATIC-SAME:     "-lpthread"


! Check powerpc64-ibm-aix 64-bit linking to shared flang-rt by option 
! RUN: %flang -shared-libflangrt -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64-ibm-aix \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefix=AIX64-LD-PER-TARGET-SHARED

! AIX64-LD-PER-TARGET-SHARED:     "-fc1" "-triple" "powerpc64-ibm-aix"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! AIX64-LD-PER-TARGET-SHARED:     "{{.*}}ld{{(.exe)?}}"
! AIX64-LD-PER-TARGET-SHARED-NOT: "-bnso"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-b64"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-bpT:0x100000000" "-bpD:0x110000000"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-lc"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-lflang_rt.runtime"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-lm"
! AIX64-LD-PER-TARGET-SHARED-SAME:     "-lpthread"


! Check powerpc64le-unknown-linux-gnu 64-bit linking to shared flang-rt by default
! RUN: %flang -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64le-unknown-linux-gnu \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefixes=LOP64-LD-PER-TARGET-DEFAULT

! LOP64-LD-PER-TARGET-DEFAULT:     "-fc1" "-triple" "powerpc64le-unknown-linux-gnu"
! LOP64-LD-PER-TARGET-DEFAULT-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! LOP64-LD-PER-TARGET-DEFAULT:     "{{.*}}ld{{(.exe)?}}"
! LOP64-LD-PER-TARGET-DEFAULT-NOT: "-bnso"
! LOP64-LD-PER-TARGET-DEFAULT-SAME:     "-lflang_rt.runtime"
! LOP64-LD-PER-TARGET-DEFAULT-SAME:     "-lm"
! LOP64-LD-PER-TARGET-DEFAULT-SAME:     "-lc"


! Check powerpc64le-unknown-linux-gnu 64-bit linking to static flang-rt by option
! RUN: %flang -static-libflangrt -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64le-unknown-linux-gnu \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefixes=LOP64-LD-PER-TARGET-STATIC

! LOP64-LD-PER-TARGET-STATIC:     "-fc1" "-triple" "powerpc64le-unknown-linux-gnu"
! LOP64-LD-PER-TARGET-STATIC-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! LOP64-LD-PER-TARGET-STATIC:     "{{.*}}ld{{(.exe)?}}"
! LOP64-LD-PER-TARGET-STATIC-NOT: "-bnso"
! LOP64-LD-PER-TARGET-STATIC-SAME:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}powerpc64le-unknown-linux-gnu{{/|\\\\}}libflang_rt.runtime.a"
! LOP64-LD-PER-TARGET-STATIC-SAME:     "-lm"
! LOP64-LD-PER-TARGET-STATIC-SAME:     "-lc"


! Check powerpc64le-unknown-linux-gnu 64-bit linking to shared flang-rt by option
! RUN: %flang -shared-libflangrt -Werror %s -### 2>&1 \
! RUN:        --target=powerpc64le-unknown-linux-gnu \
! RUN:        -resource-dir=%S/../../../clang/test/Driver/Inputs/resource_dir_with_per_target_subdir \
! RUN:   | FileCheck %s --check-prefixes=LOP64-LD-PER-TARGET-SHARED

! LOP64-LD-PER-TARGET-SHARED:     "-fc1" "-triple" "powerpc64le-unknown-linux-gnu"
! LOP64-LD-PER-TARGET-SHARED-SAME:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
! LOP64-LD-PER-TARGET-SHARED:     "{{.*}}ld{{(.exe)?}}"
! LOP64-LD-PER-TARGET-SHARED-NOT: "-bnso"
! LOP64-LD-PER-TARGET-SHARED-SAME:     "-lflang_rt.runtime"
! LOP64-LD-PER-TARGET-SHARED-SAME:     "-lm"
! LOP64-LD-PER-TARGET-SHARED-SAME:     "-lc"
