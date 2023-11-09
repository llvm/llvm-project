// This test checks that IFUNC trampoline is properly recognised by BOLT

// With -O0 indirect call is performed on IPLT trampoline. IPLT trampoline
// has IFUNC symbol.
// RUN: %clang %cflags -nostdlib -O0 -no-pie %s -fuse-ld=lld \
// RUN:    -o %t.O0.exe -Wl,-q
// RUN: llvm-bolt %t.O0.exe -o %t.O0.bolt.exe \
// RUN:   --print-disasm --print-only=_start | \
// RUN:   FileCheck --check-prefix=O0_CHECK %s

// With -O3 direct call is performed on IPLT trampoline. IPLT trampoline
// doesn't have associated symbol. The ifunc symbol has the same address as
// IFUNC resolver function.
// RUN: %clang %cflags -nostdlib -O3 %s -fuse-ld=lld -fPIC -pie \
// RUN:   -o %t.O3_pie.exe -Wl,-q
// RUN: llvm-bolt %t.O3_pie.exe -o %t.O3_pie.bolt.exe  \
// RUN:   --print-disasm --print-only=_start | \
// RUN:   FileCheck --check-prefix=O3_CHECK %s

// Check that IPLT trampoline located in .plt section are normally handled by
// BOLT. The gnu-ld linker doesn't use separate .iplt section.
// RUN: %clang %cflags -nostdlib -O3 %s -fuse-ld=lld -fPIC -pie \
// RUN:   -T %p/Inputs/iplt.ld -o %t.iplt_O3_pie.exe -Wl,-q
// RUN: llvm-bolt %t.iplt_O3_pie.exe -o %t.iplt_O3_pie.bolt.exe  \
// RUN:   --print-disasm --print-only=_start  | \
// RUN:   FileCheck --check-prefix=O3_CHECK %s

// O0_CHECK: adr x{{[0-9]+}}, ifoo
// O3_CHECK: b "{{resolver_foo|ifoo}}{{.*}}@PLT"

static void foo() {}

static void *resolver_foo(void) { return foo; }

__attribute__((ifunc("resolver_foo"))) void ifoo();

void _start() { ifoo(); }
