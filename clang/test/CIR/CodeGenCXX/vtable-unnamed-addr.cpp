// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-ELF --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM-ELF --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-ELF --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fno-rtti -fclangir -emit-cir %s -o %t-nortti.cir
// RUN: FileCheck --check-prefix=CIR-NORTTI --input-file=%t-nortti.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fno-rtti -fclangir -emit-llvm %s -o %t-nortti-cir.ll
// RUN: FileCheck --check-prefix=LLVM-NORTTI --input-file=%t-nortti-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fno-rtti -emit-llvm %s -o %t-nortti.ll
// RUN: FileCheck --check-prefix=LLVM-NORTTI --input-file=%t-nortti.ll %s

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -fclangir -emit-cir %s -o %t-darwin.cir
// RUN: FileCheck --check-prefix=CIR-DARWIN --input-file=%t-darwin.cir %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -fclangir -emit-llvm %s -o %t-darwin-cir.ll
// RUN: FileCheck --check-prefix=LLVM-DARWIN --input-file=%t-darwin-cir.ll %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -emit-llvm %s -o %t-darwin.ll
// RUN: FileCheck --check-prefix=LLVM-DARWIN --input-file=%t-darwin.ll %s

// A vtable is marked unnamed_addr when its address is insignificant: either
// no RTTI is emitted (so dynamic_cast cannot rely on its address), or it has
// weak linkage on a target (Mach-O) that may duplicate such vtables.

// Strong vtable (key function defined in this TU).
struct Key { virtual void f(); };
void Key::f() {}

// Weak (linkonce_odr) vtable.
struct Weak { virtual void g() {} };

// Construction vtables (_ZTC) come from a class with virtual bases.
struct A { virtual void a() {} };
struct B : virtual A { virtual void b() {} };
struct C : virtual A { virtual void c() {} };
struct D : B, C { void d(); };
void D::d() {}

void use() {
  Weak w;
  D x;
}

// With RTTI on ELF, vtable addresses are unique: no unnamed_addr.
// CIR-ELF-DAG: cir.global constant external @_ZTV3Key =
// CIR-ELF-DAG: cir.global constant linkonce_odr comdat @_ZTV4Weak =
// CIR-ELF-DAG: cir.global constant linkonce_odr comdat @_ZTC1D0_1B =
// LLVM-ELF-DAG: @_ZTV3Key = constant
// LLVM-ELF-DAG: @_ZTV4Weak = linkonce_odr constant
// LLVM-ELF-DAG: @_ZTC1D0_1B = linkonce_odr constant

// Without RTTI, every vtable is unnamed_addr.
// CIR-NORTTI-DAG: cir.global constant external unnamed_addr @_ZTV3Key =
// CIR-NORTTI-DAG: cir.global constant linkonce_odr comdat unnamed_addr @_ZTV4Weak =
// CIR-NORTTI-DAG: cir.global constant linkonce_odr comdat unnamed_addr @_ZTC1D0_1B =
// LLVM-NORTTI-DAG: @_ZTV3Key = unnamed_addr constant
// LLVM-NORTTI-DAG: @_ZTV4Weak = linkonce_odr unnamed_addr constant
// LLVM-NORTTI-DAG: @_ZTC1D0_1B = linkonce_odr unnamed_addr constant

// On Darwin, weak vtables may be duplicated, so they are unnamed_addr even with
// RTTI; the strong vtable keeps a unique address.
// CIR-DARWIN-DAG: cir.global constant external @_ZTV3Key =
// CIR-DARWIN-DAG: cir.global constant linkonce_odr unnamed_addr @_ZTV4Weak =
// CIR-DARWIN-DAG: cir.global constant linkonce_odr unnamed_addr @_ZTC1D0_1B =
// LLVM-DARWIN-DAG: @_ZTV3Key = constant
// LLVM-DARWIN-DAG: @_ZTV4Weak = linkonce_odr unnamed_addr constant
// LLVM-DARWIN-DAG: @_ZTC1D0_1B = linkonce_odr unnamed_addr constant
