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
// exceptions are not ignored) and "stack-protector-buffer-size".

int decl(int);

int def(int x) { return decl(x); }

// CIR: cir.func{{.*}} @def({{.*}}) -> {{.*}} attributes {{{.*}}no_trapping_math{{.*}}stack_protector_buffer_size = 8 : i32}
// CIR: cir.func private @decl({{.*}}) -> {{.*}} attributes {{{.*}}no_trapping_math{{.*}}stack_protector_buffer_size = 8 : i32}

// CIR-STRICT: cir.func{{.*}} @def({{.*}}) -> {{.*}} attributes {"cir.target-features" = "{{[^"]*}}", nothrow, optnone, stack_protector_buffer_size = 16 : i32, strictfp}
// CIR-STRICT: cir.func private @decl({{.*}}) -> {{.*}} attributes {"cir.target-features" = "{{[^"]*}}", stack_protector_buffer_size = 16 : i32}

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
