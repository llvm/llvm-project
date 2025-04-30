! XFAIL: *
! REQUIRES: classic_flang

! Check that the driver invokes flang1 correctly for preprocessed free-form
! Fortran code. Also check that the backend is invoked correctly.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -c %s -### 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-OBJECT %s
! CHECK-OBJECT: "flang1"
! CHECK-OBJECT-NOT: "-preprocess"
! CHECK-OBJECT-SAME: "-freeform"
! CHECK-OBJECT-NEXT: "flang2"
! CHECK-OBJECT-SAME: "-asm" [[LLFILE:.*.ll]]
! CHECK-OBJECT-NEXT: {{clang.* "-cc1"}}
! CHECK-OBJECT-SAME: "-o" "classic-flang.o"
! CHECK-OBJECT-SAME: "-x" "ir"
! CHECK-OBJECT-SAME: [[LLFILE]]

! Check that the driver invokes flang1 correctly when preprocessing is
! explicitly requested.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -E %s -### 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-PREPROCESS %s
! CHECK-PREPROCESS: "flang1"
! CHECK-PREPROCESS-SAME: "-preprocess"
! CHECK-PREPROCESS-SAME: "-es"
! CHECK-PREPROCESS-SAME: "-pp"
! CHECK-PREPROCESS-NOT: "flang1"
! CHECK-PREPROCESS-NOT: "flang2"
! CHECK-PREPROCESS-NOT: {{clang.* "-cc1"}}
! CHECK-PREPROCESS-NOT: {{clang.* "-cc1as"}}

! Check that the backend job (clang -cc1) is not combined into the compile job
! (flang2) even if -integrated-as is specified.

! RUN: %clang --driver-mode=flang -target x86_64-unknown-linux-gnu -integrated-as -S %s -### 2>&1 \
! RUN:   | FileCheck --check-prefix=CHECK-ASM %s
! CHECK-ASM: "flang1"
! CHECK-ASM-NEXT: "flang2"
! CHECK-ASM-SAME: "-asm" [[LLFILE:.*.ll]]
! CHECK-ASM-NEXT: {{clang.* "-cc1"}}
! CHECK-ASM-SAME: "-o" "classic-flang.s"
! CHECK-ASM-SAME: "-x" "ir"
! CHECK-ASM-SAME: [[LLFILE]]

! Check that the linker job is given the correct libraries and library paths.

! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -mp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-DYNAMIC-FLANG,CHECK-DYNAMIC-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -mp -nomp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-DYNAMIC-FLANG,CHECK-NO-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -fopenmp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-DYNAMIC-FLANG,CHECK-DYNAMIC-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -fopenmp -fno-openmp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-DYNAMIC-FLANG,CHECK-NO-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -fopenmp -static-openmp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-DYNAMIC-FLANG,CHECK-STATIC-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -fopenmp -static-flang-libs \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-FLANG,CHECK-DYNAMIC-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -static-flang-libs \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-FLANG,CHECK-NO-OMP %s

! CHECK-LD:                "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-NOT:            "-static"
! CHECK-LD:                "{{[^"]*}}classic-flang-{{[^ ]*}}.o" "-lflangmain" "-lfoo" "-L{{[^ ]*[/\\]+}}basic_linux_tree{{[/\\]+}}usr{{[/\\]+}}lib"
! CHECK-DYNAMIC-FLANG-NOT: "-Bstatic"
! CHECK-DYNAMIC-FLANG:     "-lflang" "-lflangrti" "-lpgmath" "-lpthread" "-lrt" "-lm"
! CHECK-DYNAMIC-FLANG-NOT: "-Bdynamic"
! CHECK-STATIC-FLANG:      "-Bstatic" "-lflang" "-lflangrti" "-lpgmath" "-Bdynamic" "-lpthread" "-lrt" "-lm"
! CHECK-DYNAMIC-OMP-NOT:   "-Bstatic"
! CHECK-DYNAMIC-OMP:       "-lomp" "-rpath" "{{[^ ]*[/\\]+}}basic_linux_tree{{[/\\]+}}usr{{[/\\]+}}lib"
! CHECK-DYNAMIC-OMP-NOT:   "-Bdynamic"
! CHECK-STATIC-OMP:        "-Bstatic" "-lomp" "-Bdynamic" "-rpath" "{{[^ ]*[/\\]+}}basic_linux_tree{{[/\\]+}}usr{{[/\\]+}}lib"
! CHECK-NO-OMP-NOT:        "-lomp"

! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -static -static-flang-libs \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD-STATIC,CHECK-NO-OMP %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -static -fopenmp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD-STATIC,CHECK-STATIC-BOTH %s
! RUN: %flang -target x86_64-linux-gnu -ccc-install-dir %S/../Inputs/basic_linux_tree/usr/bin -static -fopenmp -static-openmp \
! RUN:     %s -lfoo -### 2>&1 | FileCheck --check-prefixes=CHECK-LD-STATIC,CHECK-STATIC-BOTH %s
! CHECK-LD-STATIC:     "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-STATIC:     "-static" "-o" "a.out"
! CHECK-LD-STATIC:     "{{[^"]*}}classic-flang-{{[^ ]*}}.o" "-lflangmain" "-lfoo" "-L{{[^ ]*[/\\]+}}basic_linux_tree{{[/\\]+}}usr{{[/\\]+}}lib"
! CHECK-LD-STATIC-NOT: "-Bstatic"
! CHECK-LD-STATIC:     "-lflang" "-lflangrti" "-lpgmath" "-lpthread" "-lrt" "-lm"
! CHECK-LD-STATIC-NOT: "-Bdynamic"
! CHECK-STATIC-BOTH-NOT: "-Bstatic"
! CHECK-STATIC-BOTH:     "-lomp"
! CHECK-STATIC-BOTH-NOT: "-Bdynamic"
