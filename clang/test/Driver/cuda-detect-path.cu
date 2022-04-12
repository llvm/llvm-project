// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// This tests uses the PATH environment variable.
// REQUIRES: !system-windows

// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s


// Check that we follow ptxas binaries that are symlinks.
// RUN: env PATH=%S/Inputs/CUDA-symlinks/usr/bin \
// RUN:    %clang -v --target=i386-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix SYMLINKS
// RUN: env PATH=%S/Inputs/CUDA-symlinks/usr/bin \
// RUN:    %clang -v --target=i386-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix SYMLINKS
// RUN: env PATH=%S/Inputs/CUDA-symlinks/usr/bin \
// RUN:    %clang -v --target=x86_64-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix SYMLINKS
// RUN: env PATH=%S/Inputs/CUDA-symlinks/usr/bin \
// RUN:    %clang -v --target=x86_64-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix SYMLINKS


// We only take a CUDA installation from PATH if it contains libdevice.
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-unknown-linux --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-apple-macosx --sysroot=%S/no-cuda-there \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA

// We even require libdevice if -nocudalib is passed to avoid false positives
// if the distribution merges CUDA into /usr and ptxas ends up /usr/bin.
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-unknown-linux --sysroot=%S/no-cuda-there -nocudalib \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-apple-macosx --sysroot=%S/no-cuda-there -nocudalib \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-unknown-linux --sysroot=%S/no-cuda-there -nocudalib \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA-nolibdevice/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-apple-macosx --sysroot=%S/no-cuda-there -nocudalib \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA


// Check that the CUDA installation in PATH is not taken when passing
// the option --cuda-path-ignore-env.
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-unknown-linux --sysroot=%S/no-cuda-there --cuda-path-ignore-env \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=i386-apple-macosx --sysroot=%S/no-cuda-there --cuda-path-ignore-env \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-unknown-linux --sysroot=%S/no-cuda-there --cuda-path-ignore-env \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA
// RUN: env PATH=%S/Inputs/CUDA/usr/local/cuda/bin \
// RUN:    %clang -v --target=x86_64-apple-macosx --sysroot=%S/no-cuda-there --cuda-path-ignore-env \
// RUN:    2>&1 | FileCheck %s --check-prefix NOCUDA

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// SYMLINKS: Found CUDA installation: {{.*}}/Inputs/CUDA-symlinks/opt/cuda
// NOCUDA-NOT: Found CUDA installation:
