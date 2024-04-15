// RUN: %clang --target=x86_64-unknown-nacl -### %s -c -o %t.o 2>&1 | FileCheck %s -check-prefix=ECHO
// RUN: %clang -target x86_64-unknown-nacl %s -emit-llvm -S -c -o - | FileCheck %s
// RUN: %clang -target x86_64-unknown-nacl %s -emit-llvm -S -c -pthread -o - | FileCheck %s -check-prefix=THREADS

// ECHO: {{.*}} "-cc1" {{.*}}x86_64-nacl-defines.c

// Check platform defines

// CHECK: __LITTLE_ENDIAN__defined
#ifdef __LITTLE_ENDIAN__
void __LITTLE_ENDIAN__defined() {}
#endif

// CHECK: __native_client__defined
#ifdef __native_client__
void __native_client__defined() {}
#endif

// CHECK: __x86_64__defined
#ifdef __x86_64__
void __x86_64__defined() {}
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
