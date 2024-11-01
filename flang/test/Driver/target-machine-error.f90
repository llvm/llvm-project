! RUN: not %flang  --target=some-invalid-triple -S  %s -o \
! RUN:   /dev/null 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -triple some-invalid-triple -S %s -o \
! RUN:   /dev/null 2>&1 | FileCheck %s

! CHECK: error: unable to create target: 'No available targets are compatible with triple "some-invalid-triple"'
