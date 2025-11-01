// RUN: %clang -x hip %s --cuda-device-only --offload-arch=amdgcnspirv -use-experimental-spirv-backend -nogpuinc -nogpulib -ccc-print-phases 2>&1 | FileCheck %s 

// CHECK: [[P0:[0-9]+]]: input, "{{.*}}.c", hip, (device-hip, amdgcnspirv)
// CHECK: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-hip, amdgcnspirv)
// CHECK: [[P3:[0-9]+]]: backend, {[[P2]]}, spv, (device-hip, amdgcnspirv)
// CHECK: [[P4:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P3]]}, spv
// CHECK: [[P5:[0-9]+]]: linker, {[[P4]]}, hip-fatbin, (device-hip, )
// CHECK: [[P6:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa:)" {[[P5]]}, hip-fatbin
