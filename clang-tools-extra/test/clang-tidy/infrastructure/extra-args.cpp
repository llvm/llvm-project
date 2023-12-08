// RUN: clang-tidy -checks='-*,modernize-use-override' \
// RUN:   -config='{ExtraArgs: ["-DTEST4"], ExtraArgsBefore: ["-DTEST1"]}' \
// RUN:   -extra-arg=-DTEST3 -extra-arg-before=-DTEST2 %s -- -v 2>&1 \
// RUN:   | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// CHECK: {{^}}clang Invocation:{{$}}
// CHECK-NEXT: {{"-D" "TEST1" .*"-D" "TEST2" .*"-D" "TEST3" .*"-D" "TEST4"}}
