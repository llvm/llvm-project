// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fno-threadsafe-statics -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fno-threadsafe-statics -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM,LLVM-CIR
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fno-threadsafe-statics -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM,LLVM-OGCG

// On ARM-style targets (here, AArch64), function-scope static locals use the
// ARM ABI rule that only bit 0 of the guard is checked. The guard size depends
// on classic codegen's `useInt8GuardVariable = !threadsafe && hasInternalLinkage`
// (see ItaniumCXXABI::EmitGuardedInit). Both branches below are NYI in the CIR
// LoweringPrepare pass and currently fail; fill in the expected output once
// support is added.

int bar();

// `useInt8GuardVariable == true`: thread-safe statics are disabled and the
// guard for a non-inline function's static has internal linkage, so an i8
// guard byte is used.
void byte_guard() {
  static int a = bar();
}

// CIR: cir.func {{.*}} @_Z10byte_guardv()
// CIR:   %[[A:.*]] = cir.get_global static_local @_ZZ10byte_guardvE1a : !cir.ptr<!s32i>
// CIR:   %[[GUARD:.*]] = cir.get_global @_ZGVZ10byte_guardvE1a : !cir.ptr<!s8i>
// CIR:   %[[GUARD_LOAD:.*]] = cir.load{{.*}} %[[GUARD]]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0>
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_LOAD]], %[[ZERO]]
// CIR:   cir.if %[[IS_UNINIT]] {
// CIR:     %[[A:.*]] = cir.get_global static_local @_ZZ10byte_guardvE1a : !cir.ptr<!s32i>
// CIR:     %[[BAR:.*]] = cir.call @_Z3barv()
// CIR:     cir.store{{.*}} %[[BAR]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:     cir.store %[[ONE]], %[[GUARD]] : !s8i, !cir.ptr<!s8i>
// CIR:   }

// LLVM: define {{.*}} void @_Z10byte_guardv()
// LLVM:   %[[GUARD:.*]] = load i8, ptr @_ZGVZ10byte_guardvE1a
// LLVM:   %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD]], 0
// LLVM:   br i1 %[[IS_UNINIT]], label %[[DO_INIT:.*]], label %[[DONE:[^,]+]]
// LLVM: [[DO_INIT]]:
// LLVM:   %[[BAR:.*]] = call {{.*}} i32 @_Z3barv()
// LLVM:   store i32 %[[BAR]], ptr @_ZZ10byte_guardvE1a
// LLVM:   store i8 1, ptr @_ZGVZ10byte_guardvE1a
// LLVM:   br label %[[DONE]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// `useInt8GuardVariable == false`: the static inside an inline function has
// linkonce_odr linkage, so the full ARM size-typed guard (i64 on AArch64) is
// used even with -fno-threadsafe-statics.
inline int word_guard() {
  static int b = bar();
  return b;
}

// CIR: cir.func {{.*}} @_Z10word_guardv()
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[B:.*]] = cir.get_global static_local @_ZZ10word_guardvE1b : !cir.ptr<!s32i>
// CIR:   %[[GUARD:.*]] = cir.get_global @_ZGVZ10word_guardvE1b : !cir.ptr<!s64i>
// CIR:   %[[GUARD_BYTE_PTR:.*]] = cir.cast bitcast %[[GUARD]] : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR:   %[[GUARD_LOAD:.*]] = cir.load{{.*}} %[[GUARD_BYTE_PTR]] : !cir.ptr<!s8i>, !s8i
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR:   %[[GUARD_AND:.*]] = cir.and %[[GUARD_LOAD]], %[[ONE]] : !s8i
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR:   %[[IS_UNINIT:.*]] = cir.cmp eq %[[GUARD_AND]], %[[ZERO]] : !s8i
// CIR:   cir.if %[[IS_UNINIT]] {
// CIR:     %[[B_TOO:.*]] = cir.get_global static_local @_ZZ10word_guardvE1b : !cir.ptr<!s32i>
// CIR:     %[[BAR:.*]] = cir.call @_Z3barv()
// CIR:     cir.store{{.*}} %[[BAR]], %[[B_TOO]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     cir.store %[[ONE]], %[[GUARD]] : !s64i, !cir.ptr<!s64i>
// CIR:   }
// CIR:   %[[LOAD_B:.*]] = cir.load{{.*}} %[[B]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[LOAD_B]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[LOAD_RETVAL:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[LOAD_RETVAL]] : !s32i

// Note: The guard variable is a 64-bit value, but the load and truncate of
//       that value gets folded to an 8-byte-aligned i8 load on both LLVM via
//       CIR and OGCG, while the store of the value only gets folded in OGCG.

// LLVM:    define {{.*}} i32 @_Z10word_guardv()
// LLVM:      %[[GUARD:.*]] = load i8, ptr @_ZGVZ10word_guardvE1b, align 8
// LLVM:      %[[GUARD_AND:.*]] = and i8 %[[GUARD]], 1
// LLVM:      %[[IS_UNINIT:.*]] = icmp eq i8 %[[GUARD_AND]], 0
// LLVM:      br i1 %[[IS_UNINIT]], label %[[DO_INIT:.*]], label %[[DONE:[^,]+]]
// LLVM:    [[DO_INIT]]:
// LLVM:      %[[BAR:.*]] = call {{.*}} i32 @_Z3barv()
// LLVM:      store i32 %[[BAR]], ptr @_ZZ10word_guardvE1b
// LLVM-CIR:  store i64 1, ptr @_ZGVZ10word_guardvE1b, align 8
// LLVM-OGCG: store i8 1, ptr @_ZGVZ10word_guardvE1b, align 8
// LLVM:      br label %[[DONE]]
// LLVM:    [[DONE]]:
// LLVM:      %[[LOAD_B:.*]] = load i32, ptr @_ZZ10word_guardvE1b
// LLVM:      ret i32 %{{.*}}

// This is just here to trigger the emission of word_guard().
int call_word_guard() { return word_guard(); }
