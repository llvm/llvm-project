// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR --implicit-check-not=cir.cxx_module_

// CIRGen precomputes the mangled C++20 named-module initializer function
// name and stores it as a module-level attribute so LoweringPrepare can
// build the initializer without a live ASTContext after split-compilation.
// The dynamic initializer below forces LoweringPrepare to actually emit that
// initializer function, which must have external linkage for a named-module
// interface unit.

export module A;

int foo();
int x = foo();

// CIR-BEFORE-LPP: module
// CIR-BEFORE-LPP-SAME: cir.cxx_module_init_fn_name = "_ZGIW1A"

// The initializer for a named-module interface unit has external linkage.
// (Internal linkage would render as "cir.func internal private", so matching
// "cir.func private" immediately after the name asserts external linkage.)
// CIR: cir.func private @_ZGIW1A()
