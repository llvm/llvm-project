# CMake cache for a standalone (non-TheRock) build of everything needed to
# compile, run, and FileCheck the HIP device-PGO / code-coverage tests under
# compiler-rt/test/profile/{GPU,AMDGPU}.
#
# It produces, in a single configure:
#   * the host toolchain: clang, clang++, lld, llvm-profdata, llvm-cov, plus the
#     test utilities FileCheck and not (LLVM_INSTALL_UTILS);
#   * the host ROCm drain runtime clang_rt.profile_rocm (opt-in, links the
#     sanitizer interception object libs -- hence COMPILER_RT_BUILD_SANITIZERS);
#   * the amdgcn device profile runtime libclang_rt.profile.a (the baremetal
#     profile subset providing __llvm_profile_instrument_gpu and the
#     __llvm_profile_sections bounds table), built for the amdgcn-amd-amdhsa
#     runtime target via compiler-rt/cmake/caches/AMDGPU.cmake. Building the
#     device runtime requires LLVM libc for amdgcn, so libc is enabled for that
#     runtime target.
#
# Usage (see ./build.sh for a wrapper):
#   cmake -G Ninja -S llvm -B build/device-pgo \
#         -C compiler-rt/test/profile/device-pgo/toolchain-cache.cmake
#   ninja -C build/device-pgo clang lld clang-offload-bundler \
#         clang-linker-wrapper llvm-link llvm-offload-binary offload-arch \
#         llvm-profdata llvm-cov FileCheck not runtimes
#
# Outputs (under build/device-pgo):
#   bin/{clang,clang++,lld,llvm-profdata,llvm-cov,FileCheck,not}
#   lib/clang/<ver>/lib/<host-triple>/libclang_rt.profile_rocm.a
#   lib/clang/<ver>/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a

set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host;AMDGPU" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")

set(CLANG_DEFAULT_LINKER "lld" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB "compiler-rt" CACHE STRING "")

# Make FileCheck / not available in the install/bin tree for the lit-lite runner.
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

# Build host (default) and device (amdgcn) runtimes in one tree.
set(LLVM_RUNTIME_TARGETS "default;amdgcn-amd-amdhsa" CACHE STRING "")

# Host runtimes: turn on the opt-in ROCm host drain library. It pulls in the
# sanitizer interception object libs, so sanitizers must be built too.
set(RUNTIMES_default_COMPILER_RT_BUILD_PROFILE_ROCM ON CACHE BOOL "")
set(RUNTIMES_default_COMPILER_RT_BUILD_SANITIZERS ON CACHE BOOL "")

# Device runtime: the amdgcn baremetal profile subset, built with LLVM libc for
# amdgcn (freestanding C headers).
set(RUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES
  "${CMAKE_SOURCE_DIR}/../compiler-rt/cmake/caches/AMDGPU.cmake" CACHE STRING "")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "compiler-rt;libc" CACHE STRING "")
