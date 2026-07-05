// Create a dummy dxv to run
// RUN: mkdir -p %t.dir
// RUN: echo "dxv" > %t.dir/dxv && chmod 754 %t.dir/dxv

// RUN: %clang_dxc -Qstrip-rootsignature --dxv-path=%t.dir -T cs_6_0 /Fo %t.dxo -### %s 2>&1 | FileCheck %s

// Test to demonstrate that we specify to the root signature with the
// -Qstrip-rootsignature option and that it occurs before DXV

// CHECK: "{{.*}}llvm-objcopy{{(.exe)?}}" "{{.*}}.obj" "{{.*}}.obj" "--remove-section=RTS0"
// CHECK: "{{.*}}dxv{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.dxo"

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}
