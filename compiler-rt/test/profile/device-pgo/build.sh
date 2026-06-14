#!/usr/bin/env bash
# Standalone (non-TheRock) build of the toolchain + host/device runtimes used by
# the HIP device-PGO / code-coverage tests. See toolchain-cache.cmake and
# README.md for details.
#
#   ./build.sh [BUILD_DIR]
#
# Env knobs:
#   LLVM_SRC   path to the llvm-project checkout (default: repo root inferred
#              from this script's location)
#   JOBS       parallelism for ninja (default: nproc)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .../compiler-rt/test/profile/device-pgo -> repo root is four levels up.
LLVM_SRC="${LLVM_SRC:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
BUILD_DIR="${1:-${LLVM_SRC}/build/device-pgo}"
JOBS="${JOBS:-$(nproc)}"

echo "llvm-project source : ${LLVM_SRC}"
echo "build directory     : ${BUILD_DIR}"
echo "parallel jobs       : ${JOBS}"

cmake -G Ninja \
  -S "${LLVM_SRC}/llvm" \
  -B "${BUILD_DIR}" \
  -C "${SCRIPT_DIR}/toolchain-cache.cmake"

# The 'clang' target also produces the clang++ symlink. The offload toolchain
# tools (clang-offload-bundler, clang-linker-wrapper, llvm-link,
# llvm-offload-binary) and offload-arch (also installed as amdgpu-arch) are
# needed to compile/link a HIP program and to resolve --offload-arch=native /
# the multi-device test feature. 'runtimes' builds both the host (default) and
# amdgcn device runtime targets.
ninja -C "${BUILD_DIR}" -j "${JOBS}" \
  clang lld \
  clang-offload-bundler clang-linker-wrapper llvm-link llvm-offload-binary \
  offload-arch \
  llvm-profdata llvm-cov FileCheck not \
  runtimes

cat <<EOF

Build complete.

Toolchain bin : ${BUILD_DIR}/bin
Run the GPU tests with, e.g.:

  python3 ${SCRIPT_DIR}/../run_gpu_tests.py \\
      --toolchain-bin ${BUILD_DIR}/bin \\
      --hip-lib-path \${ROCM_PATH:-/opt/rocm}/lib \\
      ${SCRIPT_DIR}/../GPU ${SCRIPT_DIR}/../AMDGPU

(--toolchain-bin must be an absolute path; the runner executes RUN lines from a
temp dir. See README.md for more.)
EOF
