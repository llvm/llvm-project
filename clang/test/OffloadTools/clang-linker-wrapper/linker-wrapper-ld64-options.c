// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-apple-macosx15.0.0 \
// RUN:   --linker-path=/usr/bin/ld -lto_library /lib/libLTO.dylib \
// RUN:   -object_path_lto /tmp/lto.o -o a.out 2>&1 | FileCheck %s
// CHECK: ld{{.*}}-lto_library /lib/libLTO.dylib -object_path_lto /tmp/lto.o -o a.out
