// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s
// CHECK-LD: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"

// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -pg -pthread %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG %s
// CHECK-PG: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-PG: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "-lpthread_p" "-lc_p" "-lgcc" "{{.*}}crtend.o"
