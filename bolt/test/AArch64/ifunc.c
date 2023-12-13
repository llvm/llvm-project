// This test checks that IFUNC trampoline is properly recognised by BOLT

// With -O0 indirect call is performed on IPLT trampoline. IPLT trampoline
// has IFUNC symbol.
// RUN: %clang %cflags -nostdlib -O0 -no-pie %s -fuse-ld=lld \
// RUN:    -o %t.O0.exe -Wl,-q
// RUN: llvm-bolt %t.O0.exe -o %t.O0.bolt.exe \
// RUN:   --print-disasm --print-only=_start | \
// RUN:   FileCheck --check-prefix=O0_CHECK %s
// RUN: llvm-readelf -aW %t.O0.bolt.exe | \
// RUN:   FileCheck --check-prefix=REL_CHECK %s

// Non-pie static executable doesn't generate PT_DYNAMIC, check relocation
// is readed successfully and IPLT trampoline has been identified by bolt.
// RUN: %clang %cflags -nostdlib -O3 %s -fuse-ld=lld -no-pie \
// RUN:   -o %t.O3_nopie.exe -Wl,-q
// RUN: llvm-readelf -l %t.O3_nopie.exe | \
// RUN:   FileCheck --check-prefix=NON_DYN_CHECK %s
// RUN: llvm-bolt %t.O3_nopie.exe -o %t.O3_nopie.bolt.exe  \
// RUN:   --print-disasm --print-only=_start | \
// RUN:   FileCheck --check-prefix=O3_CHECK %s
// RUN: llvm-readelf -aW %t.O3_nopie.bolt.exe | \
// RUN:   FileCheck --check-prefix=REL_CHECK %s

// With -O3 direct call is performed on IPLT trampoline. IPLT trampoline
// doesn't have associated symbol. The ifunc symbol has the same address as
// IFUNC resolver function.
// RUN: %clang %cflags -nostdlib -O3 %s -fuse-ld=lld -fPIC -pie \
// RUN:   -o %t.O3_pie.exe -Wl,-q
// RUN: llvm-bolt %t.O3_pie.exe -o %t.O3_pie.bolt.exe  \
// RUN:   --print-disasm --print-only=_start | \
// RUN:   FileCheck --check-prefix=O3_CHECK %s
// RUN: llvm-readelf -aW %t.O3_pie.bolt.exe | \
// RUN:   FileCheck --check-prefix=REL_CHECK %s

// Check that IPLT trampoline located in .plt section are normally handled by
// BOLT. The gnu-ld linker doesn't use separate .iplt section.
// RUN: %clang %cflags -nostdlib -O3 %s -fuse-ld=lld -fPIC -pie \
// RUN:   -T %p/Inputs/iplt.ld -o %t.iplt_O3_pie.exe -Wl,-q
// RUN: llvm-bolt %t.iplt_O3_pie.exe -o %t.iplt_O3_pie.bolt.exe  \
// RUN:   --print-disasm --print-only=_start  | \
// RUN:   FileCheck --check-prefix=O3_CHECK %s
// RUN: llvm-readelf -aW %t.iplt_O3_pie.bolt.exe | \
// RUN:   FileCheck --check-prefix=REL_CHECK %s

// NON_DYN_CHECK-NOT: DYNAMIC

// O0_CHECK: adr x{{[0-9]+}}, ifoo
// O3_CHECK: b "{{resolver_foo|ifoo}}{{.*}}@PLT"

// REL_CHECK: R_AARCH64_IRELATIVE [[#%x,REL_SYMB_ADDR:]]
// REL_CHECK: [[#REL_SYMB_ADDR]] {{.*}} FUNC {{.*}} resolver_foo

static void foo() {}

static void *resolver_foo(void) { return foo; }

__attribute__((ifunc("resolver_foo"))) void ifoo();

void _start() { ifoo(); }
