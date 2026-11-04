// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Pair { long a, b; };
Pair makePair();

// The initializer is still in the cir.global ctor region when CallConvLowering
// runs, so its coercion slot has no enclosing cir.func to be placed in.
Pair g = makePair();

// The slot lands first in the block the initializer is later outlined into.
// CIR-LABEL: cir.func internal private @__cxx_global_var_init()
// CIR-NEXT:    %[[COERCE:.+]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_anon_struct>
// CIR:         %[[RET:.+]] = cir.call @_Z8makePairv() : () -> !rec_anon_struct
// CIR-NEXT:    cir.store %[[RET]], %[[COERCE]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>

// CIR: cir.func private @_Z8makePairv() -> !rec_anon_struct

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVM-NEXT:    %[[COERCE:.+]] = alloca { i64, i64 }, align 8
// LLVM-NEXT:    %[[RET:.+]] = call { i64, i64 } @_Z8makePairv()
// LLVM-NEXT:    store { i64, i64 } %[[RET]], ptr %[[COERCE]], align 8

// Both backends take the register pair back.  Classic pulls the eightbytes out
// with extractvalue where CIR reads them back through the slot.
// OGCG-LABEL: define internal void @__cxx_global_var_init()
// OGCG:         %[[RET:.+]] = call { i64, i64 } @_Z8makePairv()
// OGCG-NEXT:    %{{.+}} = extractvalue { i64, i64 } %[[RET]], 0
