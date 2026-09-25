// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s --check-prefix=LLVM

extern "C" {
  // A 'builtin' function.
  double strtod( const char *, char **);
}

const char * str = "Asdf";
const double parsed = strtod(str, nullptr);

// LLVM:  @_ZL6parsed = internal global double 0.000000e+00, align 8 
// LLVM:  @v = global ptr null, align 8

// CIR-BEFORE-LPP:  cir.global "private" internal dso_local @_ZL6parsed = ctor : !cir.double {
// CIR-BEFORE-LPP:    %[[GET_GLOB:.*]] = cir.get_global @_ZL6parsed : !cir.ptr<!cir.double>
// CIR-BEFORE-LPP:    %[[GET_BUILTIN:.*]] = cir.get_global @strtod : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>) -> !cir.double>>
// CIR-BEFORE-LPP:    %[[GET_STR:.*]] = cir.get_global @str : !cir.ptr<!cir.ptr<!s8i>>
// CIR-BEFORE-LPP:    %[[LOAD_STR:.*]] = cir.load align(8) %[[GET_STR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR-BEFORE-LPP:    %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.ptr<!s8i>>
// CIR-BEFORE-LPP:    %[[CALL:.*]] = cir.call @strtod(%[[LOAD_STR]], %[[NULL]])
// CIR-BEFORE-LPP:    cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !cir.double, !cir.ptr<!cir.double>
// CIR-BEFORE-LPP:    %{{.*}} = cir.get_global @_ZL6parsed : !cir.ptr<!cir.double>
// CIR-BEFORE-LPP:  }
// CIR-BEFORE-LPP:}

// CIR: cir.global "private" internal dso_local @_ZL6parsed = #cir.fp<0.000000e+00> : !cir.double
// CIR: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR:    %[[GET_GLOB:.*]] = cir.get_global @_ZL6parsed : !cir.ptr<!cir.double>
// CIR:    %[[GET_BUILTIN:.*]] = cir.get_global @strtod : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>) -> !cir.double>>
// CIR:    %[[GET_STR:.*]] = cir.get_global @str : !cir.ptr<!cir.ptr<!s8i>>
// CIR:    %[[LOAD_STR:.*]] = cir.load align(8) %[[GET_STR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR:    %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.ptr<!s8i>>
// CIR:    %[[CALL:.*]] = cir.call @strtod(%[[LOAD_STR]], %[[NULL]])
// CIR:    cir.store {{.*}}%[[CALL]], %[[GET_GLOB]] : !cir.double, !cir.ptr<!cir.double>
// CIR:    %{{.*}} = cir.get_global @_ZL6parsed : !cir.ptr<!cir.double>

// LLVM:  define internal void @__cxx_global_var_init() 
// LLVM:    %[[LOAD_STR:.*]] = load ptr, ptr @str, align 8
// LLVM:    %[[CALL:.*]] = call double @strtod(ptr noundef %[[LOAD_STR]], ptr noundef null)
// LLVM:    store double %[[CALL]], ptr @_ZL6parsed, align 8

extern "C" {
  using size_t = unsigned long;
extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
void *memcpy(void *a, const void *b, size_t c) {
  return __builtin_memcpy(a, b, c);
}
}

const void* v = memcpy(nullptr, nullptr, 1);
// CIR-BEFORE-LPP:  cir.global external @v = ctor : !cir.ptr<!void> {
// CIR-BEFORE-LPP:    %[[GET_V:.*]] = cir.get_global @v : !cir.ptr<!cir.ptr<!void>>
// CIR-BEFORE-LPP:    %[[NULL1:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-BEFORE-LPP:    %[[NULL2:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-BEFORE-LPP:    %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR-BEFORE-LPP:    %[[MEMCPY:.*]] = cir.call @memcpy.inline(%[[NULL1]], %[[NULL2]], %[[ONE]]) nothrow
// CIR-BEFORE-LPP:    cir.store align(8) %[[MEMCPY]], %[[GET_V]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR-BEFORE-LPP:  }

// CIR:  cir.global external @v = #cir.ptr<null> : !cir.ptr<!void> {alignment = 8 : i64, ast = #cir.var.decl.ast}
// CIR:  cir.func internal private @__cxx_global_var_init.1() {
// CIR:    %[[GET_V:.*]] = cir.get_global @v : !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[NULL1:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:    %[[NULL2:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:    %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:    %[[MEMCPY:.*]] = cir.call @memcpy.inline(%[[NULL1]], %[[NULL2]], %[[ONE]]) nothrow
// CIR:    cir.store align(8) %[[MEMCPY]], %[[GET_V]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return
// CIR:  }

// LLVM: define internal void @__cxx_global_var_init.1()
// LLVM: call ptr @memcpy.inline({{.*}}
// LLVM: store{{.*}}@v

