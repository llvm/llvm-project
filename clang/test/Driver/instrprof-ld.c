// Test instrumented profiling ld flags.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-I386 %s
//
// CHECK-LINUX-I386: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LINUX-I386: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}libclang_rt.profile-i386.a" {{.*}} "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-X86-64 %s
//
// CHECK-LINUX-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LINUX-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}libclang_rt.profile-x86_64.a" {{.*}} "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux -fprofile-instr-generate -nostdlib -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-NOSTDLIB-X86-64 %s
//
// CHECK-LINUX-NOSTDLIB-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LINUX-NOSTDLIB-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}libclang_rt.profile-x86_64.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-freebsd -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FREEBSD-X86-64 %s
//
// CHECK-FREEBSD-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-FREEBSD-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}freebsd{{/|\\\\}}libclang_rt.profile-x86_64.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -shared \
// RUN:     -target i386-unknown-linux -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-I386-SHARED %s
//
// CHECK-LINUX-I386-SHARED: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LINUX-I386-SHARED: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}libclang_rt.profile-i386.a" {{.*}} "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -shared \
// RUN:     -target x86_64-unknown-linux -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-X86-64-SHARED %s
//
// CHECK-LINUX-X86-64-SHARED: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-LINUX-X86-64-SHARED: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}libclang_rt.profile-x86_64.a" {{.*}} "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -shared \
// RUN:     -target x86_64-unknown-freebsd -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_freebsd64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FREEBSD-X86-64-SHARED %s
//
// CHECK-FREEBSD-X86-64-SHARED: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-FREEBSD-X86-64-SHARED: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}freebsd{{/|\\\\}}libclang_rt.profile-x86_64.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-apple-darwin14 -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-DARWIN-X86-64 %s
//
// CHECK-DARWIN-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DARWIN-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}darwin{{/|\\\\}}libclang_rt.profile_osx.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-apple-darwin14 -fprofile-instr-generate -nostdlib -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-DARWIN-NOSTDLIB-X86-64 %s
//
// CHECK-DARWIN-NOSTDLIB-X86-64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DARWIN-NOSTDLIB-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}darwin{{/|\\\\}}libclang_rt.profile_osx.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm64-apple-ios -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-DARWIN-ARM64 %s
//
// CHECK-DARWIN-ARM64: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-DARWIN-ARM64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}darwin{{/|\\\\}}libclang_rt.profile_ios.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv7-apple-darwin -mtvos-version-min=8.3 -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-TVOS-ARMV7 %s
//
// CHECK-TVOS-ARMV7: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-TVOS-ARMV7: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}darwin{{/|\\\\}}libclang_rt.profile_tvos.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv7s-apple-darwin10 -mwatchos-version-min=2.0 -arch armv7k -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-WATCHOS-ARMV7 %s
//
// CHECK-WATCHOS-ARMV7: "{{(.*[^-.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-WATCHOS-ARMV7: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}darwin{{/|\\\\}}libclang_rt.profile_watchos.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-pc-win32 -fprofile-instr-generate \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-WINDOWS-I386 %s
//
// CHECK-WINDOWS-I386: "{{.*}}link{{(.exe)?}}"
// CHECK-WINDOWS-I386: "{{.*}}clang_rt.profile-i386.lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-pc-win32 -fprofile-instr-generate \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-WINDOWS-X86-64 %s
//
// CHECK-WINDOWS-X86-64: "{{.*}}link{{(.exe)?}}"
// CHECK-WINDOWS-X86-64: "{{.*}}clang_rt.profile-x86_64.lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-mingw32 -fprofile-instr-generate -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=CHECK-MINGW-X86-64 %s
//
// CHECK-MINGW-X86-64: "{{(.*[^.0-9A-Z_a-z])?}}ld{{(.exe)?}}"
// CHECK-MINGW-X86-64: "{{.*}}/Inputs/resource_dir{{/|\\\\}}lib{{/|\\\\}}windows{{/|\\\\}}libclang_rt.profile-x86_64.a"

// Test instrumented profiling dependent-lib flags
//
// RUN: %clang %s -### -o %t.o -target x86_64-pc-win32 \
// RUN:     -fprofile-instr-generate 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-WINDOWS-X86-64-DEPENDENT-LIB %s
//
// CHECK-WINDOWS-X86-64-DEPENDENT-LIB: "--dependent-lib={{[^"]*}}clang_rt.profile-{{[^"]*}}.lib"
//
// RUN: %clang %s -### -o %t.o -target x86_64-mingw32 \
// RUN:     -fprofile-instr-generate 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MINGW-X86-64-DEPENDENT-LIB %s
//
// CHECK-MINGW-X86-64-DEPENDENT-LIB-NOT: "--dependent-lib={{[^"]*}}clang_rt.profile-{{[^"]*}}.a"
