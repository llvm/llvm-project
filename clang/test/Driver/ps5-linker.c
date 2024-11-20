// Test that a target emulation is supplied to the linker

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-EMU %s

// CHECK-EMU: {{ld(\.exe)?}}"
// CHECK-EMU-SAME: "-m" "elf_x86_64_fbsd"

// Test that PIE is the default for main components

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PIE %s

// CHECK-PIE: {{ld(\.exe)?}}"
// CHECK-PIE-SAME: "-pie"

// RUN: %clang --target=x86_64-sie-ps5 -no-pie %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -r %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -shared %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE,CHECK-SHARED %s
// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s

// CHECK-NO-PIE: {{ld(\.exe)?}}"
// CHECK-NO-PIE-NOT: "-pie"
// CHECK-SHARED: "--shared"

// Test the driver supplies an --image-base to the linker only for non-pie
// executables.

// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-BASE %s
// RUN: %clang --target=x86_64-sie-ps5 -no-pie %s -### 2>&1 | FileCheck --check-prefixes=CHECK-BASE %s

// CHECK-BASE: {{ld(\.exe)?}}"
// CHECK-BASE-SAME: "--image-base=0x400000"

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-BASE %s
// RUN: %clang --target=x86_64-sie-ps5 -r %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-BASE %s
// RUN: %clang --target=x86_64-sie-ps5 -shared %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-BASE %s

// CHECK-NO-BASE: {{ld(\.exe)?}}"
// CHECK-NO-BASE-NOT: --image-base

// Test the driver passes PlayStation-specific options to the linker that are
// appropriate for the type of output. Many options don't apply for relocatable
// output (-r).

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-EXE %s
// RUN: %clang --target=x86_64-sie-ps5 %s -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-EXE %s
// RUN: %clang --target=x86_64-sie-ps5 %s -static -### 2>&1 | FileCheck --check-prefixes=CHECK-EXE %s
// RUN: %clang --target=x86_64-sie-ps5 %s -r -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-EXE %s

// CHECK-EXE: {{ld(\.exe)?}}"
// CHECK-EXE-SAME: "--eh-frame-hdr"
// CHECK-EXE-SAME: "--hash-style=sysv"
// CHECK-EXE-SAME: "--build-id=uuid"
// CHECK-EXE-SAME: "--unresolved-symbols=report-all"
// CHECK-EXE-SAME: "-z" "now"
// CHECK-EXE-SAME: "-z" "start-stop-visibility=hidden"
// CHECK-EXE-SAME: "-z" "rodynamic"
// CHECK-EXE-SAME: "-z" "common-page-size=0x4000"
// CHECK-EXE-SAME: "-z" "max-page-size=0x4000"
// CHECK-EXE-SAME: "-z" "dead-reloc-in-nonalloc=.debug_*=0xffffffffffffffff"
// CHECK-EXE-SAME: "-z" "dead-reloc-in-nonalloc=.debug_ranges=0xfffffffffffffffe"
// CHECK-EXE-SAME: "-z" "dead-reloc-in-nonalloc=.debug_loc=0xfffffffffffffffe"

// CHECK-NO-EXE: {{ld(\.exe)?}}"
// CHECK-NO-EXE-NOT: "--eh-frame-hdr"
// CHECK-NO-EXE-NOT: "--hash-style
// CHECK-NO-EXE-NOT: "--build-id
// CHECK-NO-EXE-NOT: "--unresolved-symbols
// CHECK-NO-EXE-NOT: "-z"

// Test that an appropriate linker script is supplied by the driver.

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=main %s
// RUN: %clang --target=x86_64-sie-ps5 %s -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=prx %s
// RUN: %clang --target=x86_64-sie-ps5 %s -static -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=static %s
// RUN: %clang --target=x86_64-sie-ps5 %s -r -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-SCRIPT %s

// CHECK-SCRIPT: {{ld(\.exe)?}}"
// CHECK-SCRIPT-SAME: "--default-script" "[[SCRIPT]].script"

// CHECK-NO-SCRIPT: {{ld(\.exe)?}}"
// CHECK-NO-SCRIPT-NOT: "--default-script"

// Test that -static is forwarded to the linker

// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-STATIC %s

// CHECK-STATIC: {{ld(\.exe)?}}"
// CHECK-STATIC-SAME: "-static"

// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-JMC %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-JMC %s

// CHECK: -plugin-opt=-enable-jmc-instrument

// Check the default library name.
// CHECK-JMC: "--push-state" "--whole-archive" "-lSceJmc_nosubmission" "--pop-state"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s

// CHECK-DIAG: -plugin-opt=-crash-diagnostics-dir=mydumps

// Test the driver passes a sysroot to the linker. Without --sysroot, its value
// is sourced from the SDK environment variable.

// RUN: env SCE_PROSPERO_SDK_DIR=mysdk %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-SYSROOT %s
// RUN: env SCE_PROSPERO_SDK_DIR=other %clang --target=x64_64-sie-ps5 %s -### --sysroot=mysdk 2>&1 | FileCheck --check-prefixes=CHECK-SYSROOT %s

// CHECK-SYSROOT: {{ld(\.exe)?}}"
// CHECK-SYSROOT-SAME: "--sysroot=mysdk"

// Test that "." is always added to library search paths. This is long-standing
// behavior, unique to PlayStation toolchains.

// RUN: %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LDOT %s

// CHECK-LDOT: {{ld(\.exe)?}}"
// CHECK-LDOT-SAME: "-L."

// Test that <sdk-root>/target/lib is added to library search paths, if it
// exists and no --sysroot is specified.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### --sysroot=%t.dir 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s

// CHECK-NO-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-NO-TARGETLIB-NOT: "-L{{.*[/\\]}}target/lib"

// RUN: mkdir -p %t.dir/target/lib
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-TARGETLIB %s

// CHECK-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-TARGETLIB-SAME: "-L{{.*[/\\]}}target/lib"
