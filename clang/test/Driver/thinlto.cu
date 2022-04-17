// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// -flto=thin causes a switch to llvm-bc object files.
// RUN: %clangxx -ccc-print-phases -nocudainc -nocudalib -c %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir, (host-cuda)
// CHECK-COMPILE-ACTIONS-NOT: lto-bc
// CHECK-COMPILE-ACTIONS: 12: backend, {11}, lto-bc, (host-cuda)

// RUN: %clangxx -ccc-print-phases -nocudainc -nocudalib %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}thinlto.cu", cuda, (host-cuda)
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cuda-cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir, (host-cuda)
// CHECK-COMPILELINK-ACTIONS: 3: input, "{{.*}}thinlto.cu", cuda, (device-cuda, sm_{{.*}})
// CHECK-COMPILELINK-ACTIONS: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_{{.*}})
// CHECK-COMPILELINK-ACTIONS: 5: compiler, {4}, ir, (device-cuda, sm_{{.*}})
// CHECK-COMPILELINK-ACTIONS: 6: backend, {5}, assembler, (device-cuda, sm_{{.*}})
// CHECK-COMPILELINK-ACTIONS: 7: assembler, {6}, object, (device-cuda, sm_{{.*}})
// CHECK-COMPILELINK-ACTIONS: 8: offload, "device-cuda (nvptx{{.*}}-nvidia-cuda:sm_{{.*}})" {7}, object
// CHECK-COMPILELINK-ACTIONS: 9: offload, "device-cuda (nvptx{{.*}}-nvidia-cuda:sm_{{.*}})" {6}, assembler
// CHECK-COMPILELINK-ACTIONS: 10: linker, {8, 9}, cuda-fatbin, (device-cuda)
// CHECK-COMPILELINK-ACTIONS: 11: offload, "host-cuda {{.*}}" {2}, "device-cuda{{.*}}" {10}, ir
// CHECK-COMPILELINK-ACTIONS: 12: backend, {11}, lto-bc, (host-cuda)
// CHECK-COMPILELINK-ACTIONS: 13: linker, {12}, image, (host-cuda)
