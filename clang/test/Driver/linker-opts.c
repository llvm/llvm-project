// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: env LIBRARY_PATH=%t/test1 %clang -target %itanium_abi_triple %s -la -### 2>&1 | FileCheck %s
// CHECK: "-L{{.*}}/test1"
// CHECK: "{{[^"]+}}.o"
// CHECK: "-la"

// GCC driver is used as linker on cygming. It should be aware of LIBRARY_PATH.
// REQUIRES: native

// Make sure that LIBRARY_PATH works for both i386 and x86_64 on Darwin.
// RUN: env LIBRARY_PATH=%t/test1 %clang -target x86_64-apple-darwin %s -la -### 2>&1 | FileCheck %s
// RUN: env LIBRARY_PATH=%t/test1 %clang -target i386-apple-darwin %s -la -### 2>&1 | FileCheck %s
//
// Make sure that we don't warn on unused compiler arguments.
// RUN: %clang -Xclang -I. -x c %s -c -o %t/tmp.o
// RUN: %clang -### -I. -ibuiltininc -nobuiltininc -nostdinc -nostdinc++ -nostdlibinc -nogpuinc %t/tmp.o -Wno-msvc-not-found -o /dev/null 2>&1 | FileCheck /dev/null --implicit-check-not=warning:

// Make sure that we do warn in other cases.
// RUN: %clang %s -lfoo -c -o %t/tmp2.o -### 2>&1 | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning:{{.*}}unused

// Make sure -e and its aliases --entry and --entry= are properly passed on.
// RUN: %clang -### -target x86_64-unknown-linux-gnu --entry test %s 2>&1 | FileCheck --check-prefix=ENTRY %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu --entry=test %s 2>&1 | FileCheck --check-prefix=ENTRY %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -etest %s 2>&1 | FileCheck --check-prefix=ENTRY %s

// ENTRY: "-e" "test"
