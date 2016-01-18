// RUN: %clang -no-canonical-prefixes -target x86_64-pc-dragonfly %s -### 2> %t.log
// RUN: FileCheck -input-file %t.log %s

// CHECK: clang{{.*}}" "-cc1" "-triple" "x86_64-pc-dragonfly"
// CHECK: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/usr/libexec/ld-elf.so.{{.*}}" "--hash-style=both" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-L{{.*}}gcc4{{.*}}" "-rpath" "{{.*}}gcc4{{.*}}" "-lc" "-lgcc" "{{.*}}crtend.o" "{{.*}}crtn.o"


