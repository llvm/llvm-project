! RUN: %if aarch64-registered-target %{ %flang_fc1 -emit-fir -triple aarch64-unknown-linux-gnu -target-cpu aarch64 %s -o - | FileCheck %s --check-prefixes=ALL,ARMCPU %}
! RUN: %if aarch64-registered-target %{ %flang_fc1 -emit-fir -triple aarch64-unknown-linux-gnu -tune-cpu neoverse-n1 %s -o - | FileCheck %s --check-prefixes=ALL,ARMTUNE %}
! RUN: %if aarch64-registered-target %{ %flang_fc1 -emit-fir -triple aarch64-unknown-linux-gnu -target-cpu aarch64 -tune-cpu neoverse-n1 %s -o - | FileCheck %s --check-prefixes=ALL,ARMBOTH %}

! RUN: %if x86-registered-target %{ %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o - | FileCheck %s --check-prefixes=ALL,X86CPU %}
! RUN: %if x86-registered-target %{ %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -tune-cpu pentium4 %s -o - | FileCheck %s --check-prefixes=ALL,X86TUNE %}
! RUN: %if x86-registered-target %{ %flang_fc1 -emit-fir -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -tune-cpu pentium4 %s -o - | FileCheck %s --check-prefixes=ALL,X86BOTH %}

! ALL: module attributes {

! ARMCPU-SAME:      fir.target_cpu = "aarch64"
! ARMCPU-NOT:       fir.tune_cpu = "neoverse-n1"

! ARMTUNE-SAME:     fir.tune_cpu = "neoverse-n1"

! ARMBOTH-SAME: fir.target_cpu = "aarch64"
! ARMBOTH-SAME: fir.tune_cpu = "neoverse-n1"  

! X86CPU-SAME:      fir.target_cpu = "x86-64"
! X86CPU-NOT:       fir.tune_cpu = "pentium4"

! X86TUNE-SAME:     fir.tune_cpu = "pentium4"

! X86BOTH-SAME: fir.target_cpu = "x86-64"
! X86BOTH-SAME: fir.tune_cpu = "pentium4"
