// RUN: %clang %s -### -target x86_64-unknown-cloudabi 2>&1 | FileCheck %s -check-prefix=SAFESTACK
// SAFESTACK: "-cc1" "-triple" "x86_64-unknown-cloudabi" {{.*}} "-ffunction-sections" "-fdata-sections" {{.*}} "-fsanitize=safe-stack"
// SAFESTACK: "-Bstatic" "-pie" "--no-dynamic-linker" "-zrelro" "--eh-frame-hdr" "--gc-sections" "-o" "a.out" "crt0.o" "crtbegin.o" "{{.*}}" "{{.*}}" "-lc" "-lcompiler_rt" "crtend.o"

// RUN: %clang %s -### -target x86_64-unknown-cloudabi -fno-sanitize=safe-stack 2>&1 | FileCheck %s -check-prefix=NOSAFESTACK
// NOSAFESTACK: "-cc1" "-triple" "x86_64-unknown-cloudabi" {{.*}} "-ffunction-sections" "-fdata-sections"
// NOSAFESTACK-NOT: "-fsanitize=safe-stack"
// NOSAFESTACK: "-Bstatic" "-pie" "--no-dynamic-linker" "-zrelro" "--eh-frame-hdr" "--gc-sections" "-o" "a.out" "crt0.o" "crtbegin.o" "{{.*}}" "{{.*}}" "-lc" "-lcompiler_rt" "crtend.o"
