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
