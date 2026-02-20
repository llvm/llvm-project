! Test that invalid cpu is rejected.

! REQUIRES: x86-registered-target

! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -target-cpu not_valid_cpu -o - -S %s 2>&1 | FileCheck %s

! CHECK: error: unknown target CPU 'not_valid_cpu'
