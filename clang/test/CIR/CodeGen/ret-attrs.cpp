// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {
int &not_noundef_because_strict_return(){}
// CIR: cir.func{{.*}}@not_noundef_because_strict_return() -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull})
// LLVM: define dso_local nonnull align 4 dereferenceable(4) ptr @not_noundef_because_strict_return()
}

struct Struct{};

using MemPtrTy = void (Struct::*)();

MemPtrTy not_noundef_memptr(MemPtrTy t){}
// CIR: cir.func no_inline dso_local @_Z18not_noundef_memptrM6StructFvvE({{.*}}) -> !rec_anon_struct {
// LLVM: define dso_local { i64, i64 } @_Z18not_noundef_memptrM6StructFvvE({{.*}})

void not_noundef_void(){}
// CIR: cir.func no_inline dso_local @_Z16not_noundef_voidv()
// LLVM: define dso_local void @_Z16not_noundef_voidv()

int &has_noundef_ref() {}
// CIR: cir.func no_inline dso_local @_Z15has_noundef_refv() -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})
// LLVM: define dso_local noundef nonnull align 4 dereferenceable(4) ptr @_Z15has_noundef_refv()

struct Incomplete;
Incomplete &no_deref_incomplete(){}
// CIR: cir.func no_inline dso_local @_Z19no_deref_incompletev() -> (!cir.ptr<!rec_Incomplete> {llvm.align = 1 : i64, llvm.nonnull, llvm.noundef})
// LLVM: define dso_local noundef nonnull align 1 ptr @_Z19no_deref_incompletev()

// Nonnull is on ALL references unless we have a non-0 target address space, so
// this isn't really testable yet.

int &no_align_not_obj_type(){}
// CIR:  cir.func no_inline dso_local @_Z21no_align_not_obj_typev() -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})
// LLVM: define dso_local noundef nonnull align 4 dereferenceable(4) ptr @_Z21no_align_not_obj_typev()

Struct &all_attrs(){}
// CIR: cir.func no_inline dso_local @_Z9all_attrsv() -> (!cir.ptr<!rec_Struct> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef})
// LLVM: define dso_local noundef nonnull align 1 dereferenceable(1) ptr @_Z9all_attrsv()

void calls(MemPtrTy mpt) {
  not_noundef_because_strict_return();
  // CIR: cir.call @not_noundef_because_strict_return() : () -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull})
  // LLVM: call nonnull align 4 dereferenceable(4) ptr @not_noundef_because_strict_return()
  not_noundef_void();
  // CIR: cir.call @_Z16not_noundef_voidv() : () -> ()
  // LLVM: call void @_Z16not_noundef_voidv()

  not_noundef_memptr(mpt);
  // CIR: cir.call @_Z18not_noundef_memptrM6StructFvvE(%2) : (!rec_anon_struct) -> !rec_anon_struct
  // LLVM: call { i64, i64 } @_Z18not_noundef_memptrM6StructFvvE({{.*}})

  has_noundef_ref();
  // CIR: cir.call @_Z15has_noundef_refv() : () -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})
  // LLVM: call noundef nonnull align 4 dereferenceable(4) ptr @_Z15has_noundef_refv()

  no_deref_incomplete();
  // CIR: cir.call @_Z19no_deref_incompletev() : () -> (!cir.ptr<!rec_Incomplete> {llvm.align = 1 : i64, llvm.nonnull, llvm.noundef})
  // LLVM:call noundef nonnull align 1 ptr @_Z19no_deref_incompletev()

  no_align_not_obj_type();
  // CIR:  cir.call @_Z21no_align_not_obj_typev() : () -> (!cir.ptr<!s32i> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef})
  // LLVM: call noundef nonnull align 4 dereferenceable(4) ptr @_Z21no_align_not_obj_typev()

  all_attrs();
  // CIR: cir.call @_Z9all_attrsv() : () -> (!cir.ptr<!rec_Struct> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef})
  // LLVM: call noundef nonnull align 1 dereferenceable(1) ptr @_Z9all_attrsv()
}
