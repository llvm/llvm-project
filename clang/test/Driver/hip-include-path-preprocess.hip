// REQUIRES: x86-registered-target, amdgpu-registered-target
// UNSUPPORTED: system-windows

// These are preprocessing tests that verify the *content* produced by
// #include, complementing the command-line flag tests in hip-include-path.hip.

// --------------------------------------------------------------------------
// Test 1: ROCm include path must take precedence over a distro sysroot header
// of the same name.  Without the late-hook fix, the sysroot copy of
// hip/hip_runtime.h would shadow the one from --rocm-path.
// --------------------------------------------------------------------------
// RUN: rm -rf %t && mkdir -p %t/usr/include/hip %t/opt/rocm/include/hip
// RUN: echo "int SELECTED_HIP_HEADER;" > %t/opt/rocm/include/hip/hip_runtime.h
// RUN: echo "int DISTRO_HIP_HEADER;"  > %t/usr/include/hip/hip_runtime.h
// RUN: echo '#include <hip/hip_runtime.h>' > %t/test_shadow.hip
// RUN: %clang -E -P -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-host-only --offload-arch=gfx900 --hip-version=3.5 \
// RUN:   -nogpulib -nobuiltininc --sysroot=%t \
// RUN:   --rocm-path=%t/opt/rocm %t/test_shadow.hip 2>&1 \
// RUN:   | FileCheck -check-prefix=SHADOW %s

// The selected ROCm header must win over the distro sysroot header.
// SHADOW: SELECTED_HIP_HEADER
// SHADOW-NOT: DISTRO_HIP_HEADER

// --------------------------------------------------------------------------
// Test 2: #include_next chains must still work when the sysroot has a trailing
// slash (which historically doubled the system include path).  The C++ stdlib
// header cstdlib must be able to #include_next <stdlib.h> and reach the C
// system header.
// --------------------------------------------------------------------------
// RUN: rm -rf %t && mkdir -p %t/usr/include/hip %t/cxx
// RUN: echo '#include_next <stdlib.h>' > %t/cxx/cstdlib
// RUN: echo "int C_HEADER_REACHED;" > %t/usr/include/stdlib.h
// RUN: echo '#include <cstdlib>' > %t/test_include_next.hip
// RUN: %clang -E -P -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-host-only --offload-arch=gfx900 --hip-version=3.5 \
// RUN:   -nogpulib -nobuiltininc --sysroot=%t/ \
// RUN:   --rocm-path=%t/usr "-stdlib++-isystem%t/cxx" %t/test_include_next.hip 2>&1 \
// RUN:   | FileCheck -check-prefix=INCLNEXT %s

// The #include_next chain must reach the C system header.
// INCLNEXT: C_HEADER_REACHED
