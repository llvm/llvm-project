!--- Config file (full path) in output of -###
!
! RUN: %flang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-DIRS
! CHECK-DIRS: System configuration file directory: {{.*}}/Inputs/config
! CHECK-DIRS: User configuration file directory: {{.*}}/Inputs/config2
!
!--- Config file (full path) in output of -###
!
! RUN: %flang --config %S/Inputs/config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH
! RUN: %flang --config=%S/Inputs/config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH
! CHECK-HHH: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
! CHECK-HHH: -flto
!
!
!--- Config file (full path) in output of -v
!
! RUN: %flang --config %S/Inputs/config-1.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-V
! CHECK-V: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
! CHECK-V: -flto
!
!--- Config file in output of -###
!
! RUN: %flang --config-system-dir=%S/Inputs --config-user-dir= --config config-1.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-HHH2
! CHECK-HHH2: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
! CHECK-HHH2: -flto
!
!--- Config file in output of -v
!
! RUN: %flang --config-system-dir=%S/Inputs --config-user-dir= --config config-1.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-V2
! CHECK-V2: Configuration file: {{.*}}Inputs{{.}}config-1.cfg
! CHECK-V2: -flto
!
!--- Nested config files
!
! RUN: %flang --config-system-dir=%S/Inputs --config-user-dir= --config config-2.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED
! CHECK-NESTED: Configuration file: {{.*}}Inputs{{.}}config-2.cfg
! CHECK-NESTED: -fno-signed-zeros
!
! RUN: %flang --config-system-dir=%S/Inputs --config-user-dir=%S/Inputs/config --config config-6.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED2
! CHECK-NESTED2: Configuration file: {{.*}}Inputs{{.}}config-6.cfg
! CHECK-NESTED2: -fstack-arrays
!
!
! RUN: %flang --config %S/Inputs/config-2a.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTEDa
! CHECK-NESTEDa: Configuration file: {{.*}}Inputs{{.}}config-2a.cfg
! CHECK-NESTEDa: -fopenmp
!
! RUN: %flang --config-system-dir=%S/Inputs --config-user-dir= --config config-2a.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NESTED2a
! CHECK-NESTED2a: Configuration file: {{.*}}Inputs{{.}}config-2a.cfg
! CHECK-NESTED2a: -fopenmp
!
!--- User directory is searched first.
!
! RUN: %flang --config-system-dir=%S/Inputs/config --config-user-dir=%S/Inputs/config2 --config config-4.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-PRECEDENCE
! CHECK-PRECEDENCE: Configuration file: {{.*}}Inputs{{.}}config2{{.}}config-4.cfg
! CHECK-PRECEDENCE: -ffp-contract=fast
!
!--- Multiple configuration files can be specified.
! RUN: %flang --config-system-dir=%S/Inputs/config --config-user-dir= --config config-4.cfg --config %S/Inputs/config2/config-4.cfg -S %s -o /dev/null -v 2>&1 | FileCheck %s -check-prefix CHECK-TWO-CONFIGS
! CHECK-TWO-CONFIGS: Configuration file: {{.*}}Inputs{{.}}config{{.}}config-4.cfg
! CHECK-TWO-CONFIGS-NEXT: Configuration file: {{.*}}Inputs{{.}}config2{{.}}config-4.cfg
! CHECK-TWO-CONFIGS: -ffp-contract=fast
! CHECK-TWO-CONFIGS: -O3

!--- The linker input flags should be moved to the end of input list and appear only when linking.
! RUN: %flang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg %s -lmylib -Wl,foo.a -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING
! RUN: %flang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -fopenmp=libomp %s -lmylib -Wl,foo.a -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING-LIBOMP-GOES-AFTER
! RUN: %flang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING
! RUN: %flang --target=aarch64-unknown-linux-gnu --config %S/Inputs/config-l.cfg -fopenmp=libomp -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING-OPENMP
! RUN: %flang --target=x86_64-pc-windows-msvc    --config %S/Inputs/config-l.cfg %s -lmylib -Wl,foo.lib -### 2>&1 | FileCheck %s -check-prefix CHECK-LINKING-MSVC
! RUN: %flang --target=x86_64-pc-windows-msvc    --config %S/Inputs/config-l.cfg -S %s -### 2>&1 | FileCheck %s -check-prefix CHECK-NOLINKING-MSVC
! CHECK-LINKING: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-LINKING: "-ffast-math"
! CHECK-LINKING: "--as-needed" "{{.*}}-{{.*}}.o" "-lmylib" "foo.a" "-lm" "-Bstatic" "-lhappy" "-Bdynamic"
! CHECK-LINKING-LIBOMP-GOES-AFTER: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-LINKING-LIBOMP-GOES-AFTER: "-ffast-math" {{.*}}"-fopenmp"
! CHECK-LINKING-LIBOMP-GOES-AFTER: "--as-needed" "{{.*}}-{{.*}}.o" "-lmylib" "foo.a" "-lm" "-Bstatic" "-lhappy" "-Bdynamic" {{.*}}"-lomp"
! CHECK-NOLINKING: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-NOLINKING: "-ffast-math"
! CHECK-NOLINKING-NO: "-lm" "-Bstatic" "-lhappy" "-Bdynamic"
! CHECK-NOLINKING-OPENMP: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-NOLINKING-OPENMP: "-ffast-math" {{.*}}"-fopenmp"
! CHECK-NOLINKING-OPENMP-NO: "-lm" "-Bstatic" "-lhappy" "-Bdynamic" {{.}}"-lomp"
! CHECK-LINKING-MSVC: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-LINKING-MSVC: "-ffast-math"
! CHECK-LINKING-MSVC: "--as-needed" "{{.*}}-{{.*}}.o" "mylib.lib" "foo.lib" "m.lib" "-Bstatic" "happy.lib" "-Bdynamic"
! CHECK-NOLINKING-MSVC: Configuration file: {{.*}}Inputs{{.}}config-l.cfg
! CHECK-NOLINKING-MSVC: "-ffast-math"
! CHECK-NOLINKING-MSVC-NO: "m.lib" "-Bstatic" "happy.lib" "-Bdynamic"
