RUN: %flang -### -frelaxed-c-loc-checks %s 2>&1 | FileCheck %s
CHECK: "-fc1"
CHECK-SAME: "-frelaxed-c-loc-checks"
