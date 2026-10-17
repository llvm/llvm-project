// Check that -l and -syslibroot are resolved the way ld64 does for a MachO host.

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: touch %t.dir/in.o %t.dir/libfoo.dylib %t.dir/libbar.tbd
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-apple-macosx15.0.0 \
// RUN:   --linker-path=/usr/bin/ld -L%t.dir -lfoo -lbar %t.dir/in.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MACHO-LIBRARY

// MACHO-LIBRARY: ld{{.*}} {{.*}}.o -o a.out

// RUN: mkdir -p %t.dir/sdk/usr/lib && touch %t.dir/sdk/usr/lib/libbaz.tbd
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-apple-macosx15.0.0 \
// RUN:   --linker-path=/usr/bin/ld -syslibroot %t.dir/sdk -lbaz %t.dir/in.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MACHO-SYSLIBROOT

// MACHO-SYSLIBROOT: ld{{.*}} -syslibroot {{.*}}sdk {{.*}}.o -o a.out
