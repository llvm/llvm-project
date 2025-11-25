// COM: Create fatbin (executable)
// RUN: clang --offload-arch=gfx900 --offload-device-only \
// RUN:  --no-gpu-bundle-output -nogpulib -nogpuinc %s -o %t.so

// RUN: lookup-code-object %t.so 0 | FileCheck --check-prefixes=EXEC %s

// EXEC: ObjectInfo[0].isa: amdgcn-amd-amdhsa--gfx900
// EXEC: ObjectInfo[0].size: {{[1-9][0-9]*}}
// EXEC: ObjectInfo[0].offset: 0
// EXEC: ObjectInfo[1].isa: amdgcn-amd-amdhsa--gfx942
// EXEC: ObjectInfo[1].size: 0
// EXEC: ObjectInfo[1].offset: 0
// EXEC: ObjectInfo[2].isa: amdgcn-amd-amdhsa--gfx950
// EXEC: ObjectInfo[2].size: 0
// EXEC: ObjectInfo[2].offset: 0

// COM: Create offload bundle
// RUN: clang --offload-arch=gfx900,gfx942 --offload-device-only \
// RUN:   --gpu-bundle-output -nogpulib -nogpuinc \
// RUN:   %s -o %t.bundle

// RUN: lookup-code-object %t.bundle 1 | FileCheck --check-prefixes=BUNDLE %s

// BUNDLE: ObjectInfo[0].isa: amdgcn-amd-amdhsa--gfx900
// BUNDLE: ObjectInfo[0].size: {{[1-9][0-9]*}}
// BUNDLE: ObjectInfo[0].offset: {{[1-9][0-9]*}}
// BUNDLE: ObjectInfo[1].isa: amdgcn-amd-amdhsa--gfx942
// BUNDLE: ObjectInfo[1].size: {{[1-9][0-9]*}}
// BUNDLE: ObjectInfo[1].offset: {{[1-9][0-9]*}}
// BUNDLE: ObjectInfo[2].isa: amdgcn-amd-amdhsa--gfx950
// BUNDLE: ObjectInfo[2].size: 0
// BUNDLE: ObjectInfo[2].offset: 0

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
