// RUN: %clang -target mipsel-unknown-nacl -### %s -emit-llvm-only -c -o %t.o 2>&1 | FileCheck %s -check-prefix=ECHO
// RUN: %clang -target mipsel-unknown-nacl %s -emit-llvm -S -c -o - | FileCheck %s
// RUN: %clang -target mipsel-unknown-nacl %s -emit-llvm -S -c -pthread -o - | FileCheck %s -check-prefix=THREADS

// ECHO: {{.*}} "-cc1" {{.*}}mipsel-nacl-defines.c

// Check platform defines

// CHECK: _MIPSELdefined
#ifdef _MIPSEL
void _MIPSELdefined() {}
#endif

// CHECK: _mipsdefined
#ifdef _mips
void _mipsdefined() {}
#endif

// CHECK: __native_client__defined
#ifdef __native_client__
void __native_client__defined() {}
#endif

// CHECK: unixdefined
#ifdef unix
void unixdefined() {}
#endif

// CHECK: __ELF__defined
#ifdef __ELF__
void __ELF__defined() {}
#endif

// CHECK: _GNU_SOURCEdefined
#ifdef _GNU_SOURCE
void _GNU_SOURCEdefined() {}
#endif

// THREADS: _REENTRANTdefined
// CHECK: _REENTRANTundefined
#ifdef _REENTRANT
void _REENTRANTdefined() {}
#else
void _REENTRANTundefined() {}
#endif
