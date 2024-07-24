! RUN: %flang -### -c --target=x86_64 -mcmodel=large -mlarge-data-threshold=32768 %s 2>&1 | FileCheck %s
! RUN: %flang -### -c --target=x86_64 -mcmodel=large -mlarge-data-threshold=59000 %s 2>&1 | FileCheck %s --check-prefix=CHECK-59000
! RUN: %flang -### -c --target=x86_64 -mcmodel=large -mlarge-data-threshold=1048576 %s 2>&1 | FileCheck %s --check-prefix=CHECK-1M
! RUN: not %flang -c --target=x86_64 -mcmodel=large -mlarge-data-threshold=nonsense %s 2>&1 | FileCheck %s --check-prefix=INVALID
! RUN: %flang -### -c --target=x86_64 -mlarge-data-threshold=32768 %s 2>&1 | FileCheck %s --check-prefix=NO-MCMODEL
! RUN: %flang -### -c --target=x86_64 -mcmodel=small -mlarge-data-threshold=32768 %s 2>&1 | FileCheck %s --check-prefix=NO-MCMODEL
! RUN: not %flang -### -c --target=aarch64 -mcmodel=small -mlarge-data-threshold=32768 %s 2>&1 | FileCheck %s --check-prefix=NOT-SUPPORTED
  
  
! CHECK: "{{.*}}flang-new" "-fc1"
! CHECK-SAME: "-mlarge-data-threshold=32768"
! CHECK-59000: "{{.*}}flang-new" "-fc1"
! CHECK-59000-SAME: "-mlarge-data-threshold=59000"
! CHECK-1M: "{{.*}}flang-new" "-fc1"
! CHECK-1M-SAME: "-mlarge-data-threshold=1048576"
! NO-MCMODEL: 'mlarge-data-threshold=' only applies to medium and large code models
! INVALID: error: invalid value 'nonsense' in '-mlarge-data-threshold='
! NOT-SUPPORTED: error: unsupported option '-mlarge-data-threshold=' for target 'aarch64'
