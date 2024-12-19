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

// Test that an appropriate linker script is supplied by the driver, but can
// be overridden with -T.

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=main %s
// RUN: %clang --target=x86_64-sie-ps5 %s -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=prx %s
// RUN: %clang --target=x86_64-sie-ps5 %s -static -### 2>&1 | FileCheck --check-prefixes=CHECK-SCRIPT -DSCRIPT=static %s
// RUN: %clang --target=x86_64-sie-ps5 %s -r -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-SCRIPT %s
// RUN: %clang --target=x86_64-sie-ps5 %s -T custom.script -### 2>&1 | FileCheck --check-prefixes=CHECK-CUSTOM-SCRIPT --implicit-check-not "\"{{-T|--script|--default-script}}\"" %s

// CHECK-SCRIPT: {{ld(\.exe)?}}"
// CHECK-SCRIPT-SAME: "--default-script" "[[SCRIPT]].script"

// CHECK-NO-SCRIPT: {{ld(\.exe)?}}"
// CHECK-NO-SCRIPT-NOT: "--default-script"

// CHECK-CUSTOM-SCRIPT: {{ld(\.exe)?}}"
// CHECK-CUSTOM-SCRIPT-SAME: "-T" "custom.script"

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

// Test that CRT objects and libraries are supplied to the linker and can be
// omitted with -noxxx options. These switches have some interaction with
// sanitizer RT libraries. That's checked in fsanitize.c

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-MAIN-CRT,CHECK-DYNAMIC-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-SHARED-CRT,CHECK-DYNAMIC-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-CRT,CHECK-STATIC-LIBCPP,CHECK-STATIC-LIBC,CHECK-STATIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -r -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-NO-LIBS %s

// RUN: %clang --target=x86_64-sie-ps5 %s -pthread -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-PTHREAD %s

// RUN: %clang --target=x86_64-sie-ps5 %s -nostartfiles -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-DYNAMIC-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostartfiles -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-DYNAMIC-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostartfiles -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-STATIC-LIBCPP,CHECK-STATIC-LIBC,CHECK-STATIC-CORE-LIBS %s

// RUN: %clang --target=x86_64-sie-ps5 %s -nodefaultlibs -pthread -fjmc -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-MAIN-CRT,CHECK-NO-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nodefaultlibs -pthread -fjmc -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-SHARED-CRT,CHECK-NO-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nodefaultlibs -pthread -fjmc -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-CRT,CHECK-NO-LIBS %s

// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib -pthread -fjmc -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-NO-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib -pthread -fjmc -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-NO-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib -pthread -fjmc -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-NO-CRT,CHECK-NO-LIBS %s

// RUN: %clang --target=x86_64-sie-ps5 %s -nolibc -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-MAIN-CRT,CHECK-NO-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nolibc -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-SHARED-CRT,CHECK-NO-LIBC,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nolibc -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-CRT,CHECK-STATIC-LIBCPP,CHECK-NO-LIBC,CHECK-STATIC-CORE-LIBS %s

// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib++ -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-MAIN-CRT,CHECK-NO-LIBCPP,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib++ -shared -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-SHARED-CRT,CHECK-NO-LIBCPP,CHECK-DYNAMIC-CORE-LIBS %s
// RUN: %clang --target=x86_64-sie-ps5 %s -nostdlib++ -static -### 2>&1 | FileCheck --check-prefixes=CHECK-LD,CHECK-STATIC-CRT,CHECK-NO-LIBCPP,CHECK-STATIC-LIBC,CHECK-STATIC-CORE-LIBS %s

// CHECK-LD: {{ld(\.exe)?}}"
// CHECK-MAIN-CRT-SAME: "crt1.o" "crti.o" "crtbegin.o"
// CHECK-SHARED-CRT-SAME: "crti.o" "crtbeginS.o"
// CHECK-STATIC-CRT-SAME: "crt1.o" "crti.o" "crtbeginT.o"

// CHECK-NO-LIBC-NOT: "-lc{{(_stub_weak)?}}"
// CHECK-NO-LIBCPP-NOT: "-l{{c_stub_weak|stdc\+\+}}"

// CHECK-DYNAMIC-LIBC-SAME: "-lc_stub_weak"
// CHECK-DYNAMIC-CORE-LIBS-SAME: "-lkernel_stub_weak"
// CHECK-STATIC-LIBCPP-SAME: "-lstdc++"
// CHECK-STATIC-LIBC-SAME: "-lm" "-lc"
// CHECK-STATIC-CORE-LIBS-SAME: "-lcompiler_rt" "-lkernel"

// CHECK-PTHREAD-SAME: "-lpthread"

// CHECK-MAIN-CRT-SAME: "crtend.o" "crtn.o"
// CHECK-SHARED-CRT-SAME: "crtendS.o" "crtn.o"
// CHECK-STATIC-CRT-SAME: "crtend.o" "crtn.o"

// CHECK-NO-CRT-NOT: "crt{{[^"]*}}.o"
// CHECK-NO-LIBS-NOT: "-l{{[^"]*}}"

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

// Test implicit library search paths are supplied to the linker, after any
// search paths specified by the user. <sdk-root>/target/lib is implicitly
// added if it exists and no --sysroot is specified. CRT objects are found
// there. "." is always implicitly added to library search paths. This is
// long-standing behavior, unique to PlayStation toolchains.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### -Luser 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### -Luser --sysroot=%t.dir 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s

// CHECK-NO-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-NO-TARGETLIB-SAME: "-Luser"
// CHECK-NO-TARGETLIB-NOT: "-L{{.*[/\\]}}target/lib"
// CHECK-NO-TARGETLIB-SAME: "-L."

// RUN: mkdir -p %t.dir/target/lib
// RUN: touch %t.dir/target/lib/crti.o
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### -Luser 2>&1 | FileCheck --check-prefixes=CHECK-TARGETLIB %s

// CHECK-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-TARGETLIB-SAME: "-Luser"
// CHECK-TARGETLIB-SAME: "-L{{.*[/\\]}}target/lib"
// CHECK-TARGETLIB-SAME: "-L."
// CHECK-TARGETLIB-SAME: "{{.*[/\\]}}target{{/|\\\\}}lib{{/|\\\\}}crti.o"
