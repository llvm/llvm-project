// Check that GPU targets do not get the host system's default C include
// paths (e.g. /usr/include, /usr/local/include) appended to the search list.
// GPU toolchains are freestanding and manage their own include paths.
//
// Use `-v -E` on the source with `-nogpulib` to print the toolchain's
// include search list and verify the host paths are absent.

// RUN: %clang --target=amdgcn-amd-amdhsa -nogpulib -v -E -x c %s \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// RUN: %clang --target=amdgcn--amdhsa -nogpulib -v -E -x c %s \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// RUN: %clang --target=nvptx64-nvidia-cuda -nogpulib -nogpuinc -v -E -x c %s \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// CHECK-NOT: /usr/include{{$}}
// CHECK-NOT: /usr/local/include{{$}}
// CHECK:     End of search list.
