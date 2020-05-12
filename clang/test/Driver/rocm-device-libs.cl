// REQUIRES: clang-driver
// REQUIRES: amdgpu-registered-target
// REQUIRES: !system-windows

// Test flush-denormals-to-zero enabled uses oclc_daz_opt_on

// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DEFAULT,GFX900-DEFAULT,GFX900,WAVE64 %s



// Make sure the different denormal default is respected for gfx8
// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx803 \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DEFAULT,GFX803-DEFAULT,GFX803,WAVE64 %s



// Make sure the non-canonical name works
// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=fiji \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DEFAULT,GFX803-DEFAULT,GFX803,WAVE64 %s



// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   -cl-denorms-are-zero \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DAZ,GFX900,WAVE64 %s


// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx803 \
// RUN:   -cl-denorms-are-zero \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DAZ,GFX803,WAVE64 %s



// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx803 \
// RUN:   -cl-finite-math-only \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-FINITE-ONLY,GFX803,WAVE64 %s



// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx803                     \
// RUN:   -cl-fp32-correctly-rounded-divide-sqrt \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-CORRECT-SQRT,GFX803,WAVE64 %s



// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx803                     \
// RUN:   -cl-fast-relaxed-math \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-FAST-RELAXED,GFX803,WAVE64 %s



// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx803                     \
// RUN:   -cl-unsafe-math-optimizations \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-UNSAFE,GFX803,WAVE64 %s

// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx1010                    \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX1010,WAVE32 %s

// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx1011                    \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX1011,WAVE32 %s

// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx1012                    \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX1012,WAVE32 %s


// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx1010 -mwavefrontsize64  \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX1010,WAVE64 %s

// RUN: %clang -### -target amdgcn-amd-amdhsa    \
// RUN:   -x cl -mcpu=gfx1010 -mwavefrontsize64 -mno-wavefrontsize64  \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX1010,WAVE32 %s

// Ignore -mno-wavefrontsize64 without wave32 support
// RUN: %clang -### -target amdgcn-amd-amdhsa       \
// RUN:   -x cl -mcpu=gfx803  -mno-wavefrontsize64  \
// RUN:   --rocm-path=%S/Inputs/rocm-device-libs    \
// RUN:   %s \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMMON,GFX803,WAVE64 %s



// Test --hip-device-lib-path format
// RUN: %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/rocm-device-libs/amdgcn/bitcode \
// RUN:   %S/opencl.cl \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DEFAULT,GFX900-DEFAULT,GFX900,WAVE64 %s

// Test environment variable HIP_DEVICE_LIB_PATH
// RUN: env HIP_DEVICE_LIB_PATH=%S/Inputs/rocm-device-libs/amdgcn/bitcode %clang -### -target amdgcn-amd-amdhsa \
// RUN:   -x cl -mcpu=gfx900 \
// RUN:   %S/opencl.cl \
// RUN: 2>&1 | FileCheck -dump-input-on-failure --check-prefixes=COMMON,COMMON-DEFAULT,GFX900-DEFAULT,GFX900,WAVE64 %s



// COMMON: "-triple" "amdgcn-amd-amdhsa"
// COMMON-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/opencl.bc"
// COMMON-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/ocml.bc"
// COMMON-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/ockl.bc"

// GFX900-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_daz_opt_off.bc"
// GFX803-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_daz_opt_on.bc"
// GFX700-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_daz_opt_on.bc"
// COMMON-DAZ-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_daz_opt_on.bc"


// COMMON-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_unsafe_math_off.bc"
// COMMON-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_finite_only_off.bc"
// COMMON-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc"


// COMMON-FINITE-ONLY-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_unsafe_math_off.bc"
// COMMON-FINITE-ONLY-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_finite_only_on.bc"
// COMMON-FINITE-ONLY-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc"


// COMMON-CORRECT-SQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_unsafe_math_off.bc"
// COMMON-CORRECT-SQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_finite_only_off.bc"
// COMMON-CORRECT-SQRT-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc"


// COMMON-FAST-RELAXED-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_unsafe_math_on.bc"
// COMMON-FAST-RELAXED-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_finite_only_on.bc"
// COMMON-FAST-RELAXED-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc"


// COMMON-UNSAFE-MATH-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_unsafe_math_on.bc"
// COMMON-UNSAFE-MATH-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_finite_only_off.bc"
// COMMON-UNSAFE-MATH-SAME: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc"

// WAVE64: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_wavefrontsize64_on.bc"
// WAVE32: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_wavefrontsize64_off.bc"

// GFX900: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_isa_version_900.bc"
// GFX803: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_isa_version_803.bc"

kernel void func(void);
