// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-BINARY

// CHECK-SPIRV-BINARY: [[P0:[0-9]+]]: input, "[[INPUT:.*]].c", hip, (host-hip)
// CHECK-SPIRV-BINARY: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (host-hip)
// CHECK-SPIRV-BINARY: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-hip)

// CHECK-SPIRV-BINARY: [[P3:[0-9]+]]: input, "[[INPUT]].c", hip, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY: [[P4:[0-9]+]]: preprocessor,  {[[P3]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY: [[P5:[0-9]+]]: compiler,  {[[P4]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY: [[P6:[0-9]+]]: offload,  "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P5]]}, ir
// CHECK-SPIRV-BINARY: [[P7:[0-9]+]]: llvm-offload-binary, {[[P6]]}, image, (device-hip)
// CHECK-SPIRV-BINARY: [[P8:[0-9]+]]: clang-linker-wrapper, {[[P7]]}, hip-fatbin, (device-hip)

// CHECK-SPIRV-BINARY: [[P9:[0-9]+]]: offload, "host-hip (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-hip (spirv64-amd-amdhsa)" {[[P8]]}, ir
// CHECK-SPIRV-BINARY: [[P10:[0-9]+]]: backend, {[[P9]]}, assembler, (host-hip)
// CHECK-SPIRV-BINARY: [[P11:[0-9]+]]: assembler, {[[P10]]}, object, (host-hip)
// CHECK-SPIRV-BINARY: [[P12:[0-9]+]]: clang-linker-wrapper, {[[P11]]}, image, (host-hip)

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend -fgpu-rdc -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-BINARY-RDC

// CHECK-SPIRV-BINARY-RDC: [[P0:[0-9]+]]: input, "[[INPUT:.*]].c", hip, (host-hip)
// CHECK-SPIRV-BINARY-RDC: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (host-hip)
// CHECK-SPIRV-BINARY-RDC: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-hip)

// CHECK-SPIRV-BINARY-RDC: [[P3:[0-9]+]]: input, "[[INPUT]].c", hip, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-RDC: [[P4:[0-9]+]]: preprocessor,  {[[P3]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-RDC: [[P5:[0-9]+]]: compiler,  {[[P4]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-RDC: [[P6:[0-9]+]]: offload,  "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P5]]}, ir
// CHECK-SPIRV-BINARY-RDC: [[P7:[0-9]+]]: llvm-offload-binary, {[[P6]]}, image, (device-hip)

// CHECK-SPIRV-BINARY-RDC: [[P8:[0-9]+]]: offload, "host-hip (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-hip (x86_64-unknown-linux-gnu)" {[[P7]]}, ir
// CHECK-SPIRV-BINARY-RDC: [[P9:[0-9]+]]: backend, {[[P8]]}, assembler, (host-hip)
// CHECK-SPIRV-BINARY-RDC: [[P10:[0-9]+]]: assembler, {[[P9]]}, object, (host-hip)
// CHECK-SPIRV-BINARY-RDC: [[P11:[0-9]+]]: clang-linker-wrapper, {[[P10]]}, image, (host-hip)

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY

// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P0:[0-9]+]]: input, "{{.*}}.c", hip, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P3:[0-9]+]]: backend, {[[P2]]}, image, (device-hip, amdgcnspirv)
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P4:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P3]]}, image
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P5:[0-9]+]]: linker, {[[P4]]}, hip-fatbin, (device-hip)
// CHECK-SPIRV-BINARY-OFFLOAD-DEVICE-ONLY: [[P6:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa)" {[[P5]]}, none

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \ 
// RUN:         -use-spirv-backend --offload-device-only -fgpu-rdc -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC

// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: [[P0:[0-9]+]]: input, "{{.*}}.c", hip, (device-hip, amdgcnspirv)
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: [[P3:[0-9]+]]: backend, {[[P2]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC: [[P4:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P3]]}, none

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -S -fgpu-rdc -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-OFFLOAD-DEVICE-ONLY-RDC

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv \
// RUN:         -nogpuinc -nogpulib -x hip %s -save-temps \
// RUN:         -use-spirv-backend --offload-device-only -S -ccc-print-phases \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY

// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: [[P0:[0-9]+]]: input, "{{.*}}.c", hip, (device-hip, amdgcnspirv)
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, hip-cpp-output, (device-hip, amdgcnspirv)
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-hip, amdgcnspirv)
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-hip, amdgcnspirv)
// CHECK-SPIRV-TEXTUAL-OFFLOAD-DEVICE-ONLY: [[P4:[0-9]+]]: offload, "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {[[P3]]}, none
