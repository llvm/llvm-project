// REQUIRES: clang-driver
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
// CHECK-COMPILELINK-ACTIONS: 3: input, "{{.*}}thinlto.cu", cuda, (device-cuda, sm_20)
// CHECK-COMPILELINK-ACTIONS: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_20)
// CHECK-COMPILELINK-ACTIONS: 5: compiler, {4}, ir, (device-cuda, sm_20)
// CHECK-COMPILELINK-ACTIONS: 6: backend, {5}, assembler, (device-cuda, sm_20)
// CHECK-COMPILELINK-ACTIONS: 7: assembler, {6}, object, (device-cuda, sm_20)
// CHECK-COMPILELINK-ACTIONS: 8: offload, "device-cuda (nvptx{{.*}}-nvidia-cuda:sm_20)" {7}, object
// CHECK-COMPILELINK-ACTIONS: 9: offload, "device-cuda (nvptx{{.*}}-nvidia-cuda:sm_20)" {6}, assembler
// CHECK-COMPILELINK-ACTIONS: 10: linker, {8, 9}, cuda-fatbin, (device-cuda)
// CHECK-COMPILELINK-ACTIONS: 11: offload, "host-cuda {{.*}}" {2}, "device-cuda{{.*}}" {10}, ir
// CHECK-COMPILELINK-ACTIONS: 12: backend, {11}, lto-bc, (host-cuda)
// CHECK-COMPILELINK-ACTIONS: 13: linker, {12}, image, (host-cuda)

// -flto=thin should cause link using gold plugin with thinlto option,
// also confirm that it takes precedence over earlier -fno-lto and -flto=full.
// RUN: %clangxx -nocudainc -nocudalib \
// RUN:        -target x86_64-unknown-linux -### %s -flto=full -fno-lto -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-THIN-ACTION < %t %s
//
// CHECK-LINK-THIN-ACTION: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"
// CHECK-LINK-THIN-ACTION: "-plugin-opt=thinlto"

// Check that subsequent -flto=full takes precedence
// RUN: %clangxx -nocudainc -nocudalib \
// RUN:        -target x86_64-unknown-linux -### %s -flto=thin -flto=full 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-FULL-ACTION < %t %s
//
// CHECK-LINK-FULL-ACTION: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"
// CHECK-LINK-FULL-ACTION-NOT: "-plugin-opt=thinlto"

// Check that subsequent -fno-lto takes precedence
// RUN: %clangxx -nocudainc -nocudalib \
// RUN:        -target x86_64-unknown-linux -### %s -flto=thin -fno-lto 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-NOLTO-ACTION < %t %s
//
// CHECK-LINK-NOLTO-ACTION-NOT: "-plugin" "{{.*}}{{[/\\]}}LLVMgold.{{dll|dylib|so}}"
// CHECK-LINK-NOLTO-ACTION-NOT: "-plugin-opt=thinlto"
