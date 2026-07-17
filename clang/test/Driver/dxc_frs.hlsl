// RUN: %clang_dxc -Vd -T cs_6_0 /Fo %t.dxo /Frs %t.rs.dxo -### %s 2>&1 | FileCheck %s

// Test to demonstrate extracting the root signature to the specified
// output file with /Frs.

// CHECK: "{{.*}}llvm-objcopy{{(.exe)?}}" "{{.*}}.obj" "{{.*}}.dxo" "--extract-section=RTS0={{.*}}.rs.dxo"
[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}
