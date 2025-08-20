// RUN: %clang_dxc -Qstrip-rootsignature -T cs_6_0 /Fo %t -### %s 2>&1 | FileCheck %s

// Test to demonstrate that we specify to the root signature with the
// -Qstrip-rootsignature option

// CHECK: "{{.*}}llvm-objcopy{{.*}}" "{{.*}}" "{{.*}}" "--remove-section=RTS0"

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}
