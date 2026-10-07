// RUN: %clang_cc1 -std=c++20 -triple x86_64-pc-windows-msvc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR --implicit-check-not=_ZGIW

// Classic codegen only enables C++20 module initializers for the Itanium
// mangler, because no Microsoft mangling for them has been settled on yet (see
// CXX20ModuleInits in CodeGenModule.cpp).  A named module built for the
// Microsoft C++ ABI therefore gets the ordinary `_GLOBAL__sub_I_` initializer
// with internal linkage, exactly as a non-modular translation unit would, and
// CIRGen matches that by only emitting the cir.cxx_module_init_fn_name
// attribute for the Itanium mangler.
//
// LoweringPrepare used to rediscover the named module from the AST whenever
// that attribute was absent, and mangled the name with an unguarded
// cast<ItaniumMangleContext>, which asserted for this target.  The attribute is
// now the only channel through which the named-module initializer reaches
// lowering, so this compiles rather than crashing.
//
// For reference, classic codegen emits for this input:
//   @llvm.global_ctors = ... { i32 65535, ptr @_GLOBAL__sub_I_<file>, ptr null }
//   define internal void @"??__Ex@@YAXXZ"() { %call = call @"?foo@@YAHXZ"()
//                                             store i32 %call, ptr @"?x@@3HA" }
//   define internal void @_GLOBAL__sub_I_<file>() { call void @"??__Ex@@YAXXZ"() }
// CIR does not yet apply the Microsoft dynamic-initializer mangling to the
// per-variable initializer, naming it __cxx_global_var_init instead; that gap
// is unrelated to named modules and reproduces for a non-modular TU too.

export module A;

int foo();
int x = foo();

// The Itanium-only attribute must not be emitted for the Microsoft mangler.
// CIR-NOT: cir.cxx_module_init_fn_name

// CIR: cir.global_ctors = [#cir.global_ctor<"_GLOBAL__sub_I_{{.*}}", 65535>]

// CIR:      cir.func internal private @__cxx_global_var_init()
// CIR:        %[[X:.*]] = cir.get_global @"?x@@3HA"
// CIR:        %[[CALL:.*]] = cir.call @"?foo@@YAHXZ"()
// CIR:        cir.store align(4) %[[CALL]], %[[X]]

// The fallback initializer has internal linkage, unlike the external-linkage
// initializer a named-module interface unit gets under the Itanium mangler, and
// it just calls the per-variable initializer.
// CIR:      cir.func internal private @_GLOBAL__sub_I_
// CIR-NEXT:   cir.call @__cxx_global_var_init()
