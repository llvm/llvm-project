// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD %s
// CHECK-LD: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"

// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -pg -pthread %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG %s
// CHECK-PG: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-PG: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}gcrt0.o" "{{.*}}crtbegin.o" "{{.*}}.o" "-lgcc" "-lpthread_p" "-lc_p" "-lgcc" "{{.*}}crtend.o"

// Check CPU type for MIPS64
// RUN: %clang -target mips64-unknown-openbsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64-CPU %s
// RUN: %clang -target mips64el-unknown-openbsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64EL-CPU %s
// CHECK-MIPS64-CPU: "-target-cpu" "mips3"
// CHECK-MIPS64EL-CPU: "-target-cpu" "mips3"

// Check that the new linker flags are passed to OpenBSD
// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -r %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-R %s
// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -s %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-S %s
// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -t %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-T %s
// RUN: %clang -no-canonical-prefixes -target i686-pc-openbsd -Z %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-Z %s
// RUN: %clang -no-canonical-prefixes -target mips64-unknown-openbsd %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-LD %s
// RUN: %clang -no-canonical-prefixes -target mips64el-unknown-openbsd %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-LD %s
// CHECK-LD-R: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD-R: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "-r" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
// CHECK-LD-S: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD-S: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "-s" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
// CHECK-LD-T: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD-T: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "-t" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
// CHECK-LD-Z: clang{{.*}}" "-cc1" "-triple" "i686-pc-openbsd"
// CHECK-LD-Z: ld{{.*}}" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "-Z" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
// CHECK-MIPS64-LD: clang{{.*}}" "-cc1" "-triple" "mips64-unknown-openbsd"
// CHECK-MIPS64-LD: ld{{.*}}" "-EB" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"
// CHECK-MIPS64EL-LD: clang{{.*}}" "-cc1" "-triple" "mips64el-unknown-openbsd"
// CHECK-MIPS64EL-LD: ld{{.*}}" "-EL" "-e" "__start" "--eh-frame-hdr" "-Bdynamic" "-dynamic-linker" "{{.*}}ld.so" "-o" "a.out" "{{.*}}crt0.o" "{{.*}}crtbegin.o" "-L{{.*}}" "{{.*}}.o" "-lgcc" "-lc" "-lgcc" "{{.*}}crtend.o"

// Check passing options to the assembler for various OpenBSD targets
// RUN: %clang -target amd64-pc-openbsd -m32 -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-AMD64-M32 %s
// RUN: %clang -target powerpc-unknown-openbsd -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-POWERPC %s
// RUN: %clang -target sparc-unknown-openbsd -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARC %s
// RUN: %clang -target sparc64-unknown-openbsd -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SPARC64 %s
// RUN: %clang -target mips64-unknown-openbsd -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64 %s
// RUN: %clang -target mips64-unknown-openbsd -fPIC -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64-PIC %s
// RUN: %clang -target mips64el-unknown-openbsd -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64EL %s
// RUN: %clang -target mips64el-unknown-openbsd -fPIC -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64EL-PIC %s
// CHECK-AMD64-M32: as{{.*}}" "--32"
// CHECK-POWERPC: as{{.*}}" "-mppc" "-many"
// CHECK-SPARC: as{{.*}}" "-32"
// CHECK-SPARC64: as{{.*}}" "-64" "-Av9a"
// CHECK-MIPS64: as{{.*}}" "-mabi" "64" "-EB"
// CHECK-MIPS64-PIC: as{{.*}}" "-mabi" "64" "-EB" "-KPIC"
// CHECK-MIPS64EL: as{{.*}}" "-mabi" "64" "-EL"
// CHECK-MIPS64EL-PIC: as{{.*}}" "-mabi" "64" "-EL" "-KPIC"
