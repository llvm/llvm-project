// RUN: %clang --target=x86_64-pc-dragonfly -### %s 2> %t.log
// RUN: FileCheck -input-file %t.log %s

// CHECK: "-cc1" "-triple" "x86_64-pc-dragonfly"
// CHECK: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/usr/libexec/ld-elf.so.{{.*}}" "--hash-style=gnu" "--enable-new-dtags" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-L{{.*}}gcc{{.*}}" "-rpath" "{{.*}}gcc{{.*}}" "-lc" "-lgcc" "{{.*}}crtend.o" "{{.*}}crtn.o"

// -r suppresses -dynamic-linker, default -l and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=x86_64-pc-dragonfly -r \
// RUN:   2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE:     "-r"
// RELOCATABLE-NOT: "-dynamic-linker"
// RELOCATABLE-NOT: "-l
// RELOCATABLE-NOT: {{.*}}crt{{[^./]+}}.o
