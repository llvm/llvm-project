// Test clang changes for NaCl Support including:
//    include paths, library paths, emulation, default static
//
// RUN: %clang -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target i686-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-I686 %s
// CHECK-I686: {{.*}}clang{{.*}}" "-cc1"
// CHECK-I686: "-fuse-init-array"
// CHECK-I686: "-target-cpu" "pentium4"
// CHECK-I686: "-resource-dir" "foo"
// CHECK-I686: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-I686: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}i686-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-I686: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include"
// CHECK-I686: as{{(.exe)?}}" "--32"
// CHECK-I686: ld{{(.exe)?}}"
// CHECK-I686: "--build-id"
// CHECK-I686: "-m" "elf_i386_nacl"
// CHECK-I686: "-static"
// CHECK-I686: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}lib32"
// CHECK-I686: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}i686-nacl{{/|\\\\}}usr{{/|\\\\}}lib"
// CHECK-I686: "-Lfoo{{/|\\\\}}lib{{/|\\\\}}i686-nacl"
// CHECK-I686-NOT: -lpthread
//
// RUN: %clang -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target x86_64-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-x86_64 %s
// CHECK-x86_64: {{.*}}clang{{.*}}" "-cc1"
// CHECK-x86_64: "-fuse-init-array"
// CHECK-x86_64: "-target-cpu" "x86-64"
// CHECK-x86_64: "-resource-dir" "foo"
// CHECK-x86_64: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-x86_64: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-x86_64: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include"
// CHECK-x86_64: as{{(.exe)?}}" "--64"
// CHECK-x86_64: ld{{(.exe)?}}"
// CHECK-x86_64: "--build-id"
// CHECK-x86_64: "-m" "elf_x86_64_nacl"
// CHECK-x86_64: "-static"
// CHECK-x86_64: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}lib"
// CHECK-x86_64: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}usr{{/|\\\\}}lib"
// CHECK-x86_64: "-Lfoo{{/|\\\\}}lib{{/|\\\\}}x86_64-nacl"
// CHECK-X86_64-NOT: -lpthread
//
// RUN: %clang -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target armv7a-unknown-nacl-gnueabihf -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARM: "-fuse-init-array"
// CHECK-ARM: "-target-cpu" "generic"
// CHECK-ARM: "-target-abi" "aapcs-linux"
// CHECK-ARM: "-mfloat-abi" "hard"
// CHECK-ARM: "-resource-dir" "foo"
// CHECK-ARM: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-ARM: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-ARM: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}include"
// CHECK-ARM: as{{(.exe)?}}"
// CHECK-ARM: "-mfloat-abi=hard"
// CHECK-ARM: ld{{(.exe)?}}"
// CHECK-ARM: "--build-id"
// CHECK-ARM: "-m" "armelf_nacl"
// CHECK-ARM: "-static"
// CHECK-ARM: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}lib"
// CHECK-ARM: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}usr{{/|\\\\}}lib"
// CHECK-ARM: "-Lfoo{{/|\\\\}}lib{{/|\\\\}}arm-nacl"
// CHECK-ARM-NOT: -lpthread
//
// RUN: %clang -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target mipsel-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPS: "-fuse-init-array"
// CHECK-MIPS: "-target-cpu" "mips32r2"
// CHECK-MIPS: "-target-abi" "o32"
// CHECK-MIPS: "-mfloat-abi" "hard"
// CHECK-MIPS: "-resource-dir" "foo"
// CHECK-MIPS: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-MIPS: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-MIPS: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}include"
// CHECK-MIPS-NOT: as{{(.exe)?}}"
// CHECK-MIPS: ld{{(.exe)?}}"
// CHECK-MIPS: "--build-id"
// CHECK-MIPS: "-m" "mipselelf_nacl"
// CHECK-MIPS: "-static"
// CHECK-MIPS: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}lib"
// CHECK-MIPS: "-L{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}usr{{/|\\\\}}lib"
// CHECK-MIPS: "-Lfoo{{/|\\\\}}lib{{/|\\\\}}mipsel-nacl"
// CHECK-MIPS: "-lpnacl_legacy"
// CHECK-MIPS-NOT: "-lpthread"

// Check that even when the target arch is just "arm" (as will be the case when
// it is inferred from the binary name) that we get the right ABI flags
// RUN: %clang -no-canonical-prefixes -### -o %t.o %s 2>&1 \
// RUN:     -target arm-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-NOV7 %s
// CHECK-ARM-NOV7: "-triple" "armv7--nacl-gnueabihf"
// CHECK-ARM-NOV7: "-target-abi" "aapcs-linux"
// CHECK-ARM-NOV7: "-mfloat-abi" "hard"
// CHECK-ARM-NOV7: as{{(.exe)?}}"
// CHECK-ARM-NOV7: "-mfloat-abi=hard"

// Test clang c++ include dirs and link line when using clang++

// RUN: %clangxx -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target armv7a-unknown-nacl-gnueabihf -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-CXX %s
// CHECK-ARM-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARM-CXX: "-resource-dir" "foo"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-ARM-CXX: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}arm-nacl{{/|\\\\}}include"
// CHECK-ARM-CXX: "-lpthread"

// RUN: %clangxx -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target i686-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-I686-CXX %s
// CHECK-I686-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-I686-CXX: "-resource-dir" "foo"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-I686-CXX: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}i686-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include"
// CHECK-I686-CXX: "-lpthread"

// RUN: %clangxx -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target x86_64-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-x86_64-CXX %s
// CHECK-x86_64-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-x86_64-CXX: "-resource-dir" "foo"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-x86_64-CXX: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}x86_64-nacl{{/|\\\\}}include"
// CHECK-x86_64-CXX: "-lpthread"

// RUN: %clangxx -no-canonical-prefixes -### -o %t.o %s \
// RUN:     -target mipsel-unknown-nacl -resource-dir foo 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-CXX %s
// CHECK-MIPS-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-MIPS-CXX: "-resource-dir" "foo"
// CHECK-MIPS-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK-MIPS-CXX: "-internal-isystem" "foo{{/|\\\\}}include"
// CHECK-MIPS-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK-MIPS-CXX: "-internal-isystem" "{{.*}}{{/|\\\\}}..{{/|\\\\}}mipsel-nacl{{/|\\\\}}include"
// CHECK-MIPS-CXX: "-lnacl"
// CHECK-MIPS-CXX: "-lpthread"
