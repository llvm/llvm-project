! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o - | FileCheck %s --check-prefixes=ALL,CPU
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -tune-cpu pentium4 %s -o - | FileCheck %s --check-prefixes=ALL,TUNE
! RUN: %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -tune-cpu pentium4 %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL: module attributes {

! CPU-SAME:      fir.target_cpu = "x86-64"
! CPU-NOT:       fir.tune_cpu = "pentium4"
  
! TUNE-SAME:     fir.tune_cpu = "pentium4"

! BOTH-SAME: fir.target_cpu = "x86-64"
! BOTH-SAME: fir.tune_cpu = "pentium4"  
