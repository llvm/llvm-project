// RUN: %clang -target xcore %s -g -Wl,L1Arg,L2Arg -Wa,A1Arg,A2Arg -fverbose-asm -v -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -target xcore -x c++ %s -g -Wl,L1Arg,L2Arg -Wa,A1Arg,A2Arg -fverbose-asm -v -### -o %t.o 2>&1 | FileCheck %s
// RUN: %clang -target xcore -x c++ %s -fexceptions -### -o %t.o 2>&1 | FileCheck -check-prefix CHECK-EXCEP %s
// RUN: %clang -target xcore %s -g0 -### -o %t.o 2>&1 | FileCheck -check-prefix CHECK-G0 %s

// CHECK: "-nostdsysteminc"
// CHECK: "-momit-leaf-frame-pointer"
// CHECK-NOT: "-mdisable-fp-elim"
// CHECK: "-fno-signed-char"
// CHECK: "-fno-use-cxa-atexit"
// CHECK-NOT: "-fcxx-exceptions"
// CHECK-NOT: "-fexceptions"
// CHECK: "-fno-common"
// CHECK: xcc" "-o"
// CHECK-NOT: "-fexceptions"
// CHECK: "-c" "-v" "-g" "-fverbose-asm" "A1Arg" "A2Arg"
// CHECK: xcc" "-o"
// CHECK-NOT: "-fexceptions"
// CHECK: "-v"
// CHECK: "L1Arg" "L2Arg"

// CHECK-EXCEP: "-fno-use-cxa-atexit"
// CHECK-EXCEP: "-fcxx-exceptions"
// CHECK-EXCEP: "-fexceptions"
// CHECK-EXCEP: "-fno-common"
// CHECK-EXCEP: xcc" "-o"
// CHECK-EXCEP-NOT: "-fexceptions"
// CHECK-EXCEP: xcc" "-o"
// CHECK-EXCEP: "-fexceptions"

// CHECK-G0: xcc"
// CHECK-G0-NOT: "-g"
// CHECK-G0: xcc"

