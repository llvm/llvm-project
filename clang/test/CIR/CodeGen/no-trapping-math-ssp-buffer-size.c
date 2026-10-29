// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -stack-protector-buffer-size 16 -fclangir -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --input-file=%t-strict.cir %s -check-prefix=CIR-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -stack-protector-buffer-size 16 -fclangir -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck --input-file=%t-strict-cir.ll %s -check-prefix=LLVM-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -stack-protector-buffer-size 16 -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s -check-prefix=LLVM-STRICT

// Both the definition and the declaration get "no-trapping-math" (unless FP
// exceptions are not ignored) and "stack-protector-buffer-size". Their values
// are TU-wide and live in #cir.lowering_lang_options, and LowerToLLVM adds them
// to every function that is not marked runtime_func.

int decl(int);

int def(int x) { return decl(x); }

// CIR: cir.lowering_lang_options = #cir.lowering_lang_options<{{.*}}no_trapping_math = true, ssp_buffer_size = 8>
// CIR-NOT: runtime_func

// CIR-STRICT: cir.lowering_lang_options = #cir.lowering_lang_options<{{.*}}no_trapping_math = false, ssp_buffer_size = 16>

// LLVM: define{{.*}} i32 @def(i32 {{.*}}) #[[DEF_ATTR:[0-9]+]]
// LLVM: declare{{.*}} i32 @decl(i32 {{.*}}) #[[DECL_ATTR:[0-9]+]]
// LLVM: attributes #[[DEF_ATTR]] = {{{.*}}"no-trapping-math"="true" "stack-protector-buffer-size"="8"
// LLVM: attributes #[[DECL_ATTR]] = {{{.*}}"no-trapping-math"="true" "stack-protector-buffer-size"="8"

// LLVM-STRICT: define{{.*}} i32 @def(i32 {{.*}}) #[[DEF_ATTR:[0-9]+]]
// LLVM-STRICT: declare{{.*}} i32 @decl(i32 {{.*}}) #[[DECL_ATTR:[0-9]+]]
// String attributes are sorted, so "no-trapping-math" would precede
// "stack-protector-buffer-size".
// LLVM-STRICT: attributes #[[DEF_ATTR]] = { {{[^"]*}}{{("min-legal-vector-width"="0" )?}}"stack-protector-buffer-size"="16"
// LLVM-STRICT: attributes #[[DECL_ATTR]] = { "stack-protector-buffer-size"="16"
