// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies C++ records on ordinary AMDGPU
// functions, where C++ rules on triviality and layout change the result.

struct Empty {};

struct NonTrivialDtor {
  int i;
  ~NonTrivialDtor();
};

struct NonCopyable {
  int i;
  NonCopyable();
  NonCopyable(const NonCopyable &) = delete;
};

struct TrivialBase { int i; };
struct DerivedSingle : TrivialBase {};
struct DerivedPair : TrivialBase { int j; };
struct MemberPtrHolder { int TrivialBase::*p; };
typedef void (TrivialBase::*mfp_t)();

// A record that cannot be passed in registers is built directly in a generic
// sret.
NonTrivialDtor ret_non_trivial_dtor() { return {1}; }

// CIR: cir.func {{.*}}@_Z20ret_non_trivial_dtorv(%[[RET:[^:]+]]: !cir.ptr<!rec_NonTrivialDtor> {{.*}}llvm.sret = !rec_NonTrivialDtor
// CIR-NEXT:   %[[I:.*]] = cir.get_member %[[RET]][0] {name = "i"} : !cir.ptr<!rec_NonTrivialDtor> -> !cir.ptr<!s32i>
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   cir.store align(4) %[[ONE]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.return
// LLVM: define {{.*}}void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind noalias writable sret(%struct.NonTrivialDtor) align 4 %[[RET:.*]])
// LLVM:   %[[I:.*]] = getelementptr inbounds nuw %struct.NonTrivialDtor, ptr %[[RET]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[I]], align 4

// The caller's local is built in place by the callee.
void use_non_trivial_dtor() { NonTrivialDtor x = ret_non_trivial_dtor(); }

// CIR: cir.func {{.*}}@_Z20use_non_trivial_dtorv()
// CIR:   %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)>
// CIR:   %[[X_VIEW:.*]] = cir.cast address_space %[[X]] : !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)> -> !cir.ptr<!rec_NonTrivialDtor>
// CIR:   cir.call @_Z20ret_non_trivial_dtorv(%[[X_VIEW]]) {{.*}}: (!cir.ptr<!rec_NonTrivialDtor> {{.*}}llvm.sret = !rec_NonTrivialDtor
// CIR:   cir.call @_ZN14NonTrivialDtorD1Ev(%[[X_VIEW]])
// LLVM: define {{.*}}void @_Z20use_non_trivial_dtorv()
// LLVM:   call void @_Z20ret_non_trivial_dtorv(ptr dead_on_unwind writable sret(%struct.NonTrivialDtor) align 4 %{{.*}})

NonCopyable ret_non_copyable() { return NonCopyable(); }

// CIR: cir.func {{.*}}@_Z16ret_non_copyablev(%arg0: !cir.ptr<!rec_NonCopyable> {{.*}}llvm.sret = !rec_NonCopyable
// LLVM: define {{.*}}void @_Z16ret_non_copyablev(ptr dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %{{.*}})

// A base subobject holding the only field still counts as a single element.
DerivedSingle ret_derived_single() { return {}; }

// CIR: cir.func {{.*}}@_Z18ret_derived_singlev() -> !s32i
// LLVM: define {{.*}}i32 @_Z18ret_derived_singlev()

void arg_derived_single(DerivedSingle d) {}

// CIR: cir.func {{.*}}@_Z18arg_derived_single13DerivedSingle(%arg0: !s32i{{.*}})
// LLVM: define {{.*}}void @_Z18arg_derived_single13DerivedSingle(i32 %{{.*}})

// A base plus a field is packed into a pair of i32 registers.
DerivedPair ret_derived_pair() { return {}; }

// CIR: cir.func {{.*}}@_Z16ret_derived_pairv() -> !cir.array<!u32i x 2>
// CIR:   %[[RET:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>, !cir.array<!u32i x 2>
// CIR:   cir.return %[[RET]] : !cir.array<!u32i x 2>
// LLVM: define {{.*}}[2 x i32] @_Z16ret_derived_pairv()
// LLVM:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// LLVM:   ret [2 x i32] %[[RET]]

void arg_derived_pair(DerivedPair d) {}

// CIR: cir.func {{.*}}@_Z16arg_derived_pair11DerivedPair(%[[D:[^:]+]]: !cir.array<!u32i x 2>{{.*}})
// CIR-NEXT:   %[[D_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[D]], %[[D_COERCE]] : !cir.array<!u32i x 2>, !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// LLVM: define {{.*}}void @_Z16arg_derived_pair11DerivedPair([2 x i32] %[[D:[^)]+]])
// LLVM:   store [2 x i32] %[[D]], ptr addrspace(5) %{{.*}}, align 4

// A data member pointer is lowered to an i64 offset before classification.
MemberPtrHolder ret_member_ptr() { return {nullptr}; }

// CIR: cir.func {{.*}}@_Z14ret_member_ptrv() -> !s64i
// LLVM: define {{.*}}i64 @_Z14ret_member_ptrv()

void arg_member_ptr(MemberPtrHolder h) {}

// CIR: cir.func {{.*}}@_Z14arg_member_ptr15MemberPtrHolder(%arg0: !s64i{{.*}})
// LLVM: define {{.*}}void @_Z14arg_member_ptr15MemberPtrHolder(i64 %{{.*}})

// A member function pointer is lowered to a pair of i64 before
// classification, and the pair passes one field per register.
void arg_mfp(mfp_t f) {}

// CIR: cir.func {{.*}}@_Z7arg_mfpM11TrivialBaseFvvE(%[[PTR:[^:]+]]: !s64i{{.*}}, %[[ADJ:[^:]+]]: !s64i{{.*}})
// CIR-NEXT:   %[[F_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_anon_struct, target_address_space(5)>
// CIR-NEXT:   %[[F0:.*]] = cir.get_member %[[F_COERCE]][0]
// CIR-NEXT:   cir.store %[[PTR]], %[[F0]]
// CIR-NEXT:   %[[F1:.*]] = cir.get_member %[[F_COERCE]][1]
// CIR-NEXT:   cir.store %[[ADJ]], %[[F1]]
// LLVM: define {{.*}}void @_Z7arg_mfpM11TrivialBaseFvvE(i64 %[[PTR:[^,]+]], i64 %[[ADJ:[^)]+]])
// LLVM:   store i64 %[[PTR]], ptr addrspace(5) %{{.*}}, align 8
// LLVM:   store i64 %[[ADJ]], ptr addrspace(5) %{{.*}}, align 8

mfp_t ret_mfp() { return nullptr; }

// CIR: cir.func {{.*}}@_Z7ret_mfpv() -> !rec_anon_struct
// CIR:   %[[RET:.*]] = cir.load %{{.*}} : !cir.ptr<!rec_anon_struct, target_address_space(5)>, !rec_anon_struct
// CIR-NEXT:   cir.return %[[RET]] : !rec_anon_struct
// LLVM: define {{.*}}{ i64, i64 } @_Z7ret_mfpv()

// An empty C++ record is one byte in size but is still dropped.
void arg_empty(Empty e) {}

// CIR: cir.func {{.*}}@_Z9arg_empty5Empty()
// LLVM: define {{.*}}void @_Z9arg_empty5Empty()

// A non-trivial record argument is passed through a pointer in addrspace(5).
void arg_non_trivial_dtor(NonTrivialDtor n) {}

// CIR: cir.func {{.*}}@_Z20arg_non_trivial_dtor14NonTrivialDtor(%[[N:[^:]+]]: !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NEXT:   %{{.*}} = cir.cast address_space %[[N]] : !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)> -> !cir.ptr<!rec_NonTrivialDtor>
// CIR-NEXT:   cir.return
// LLVM: define {{.*}}void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr addrspace(5) nofreeobj noundef align 4 dereferenceable(4) %{{.*}})

// The caller passes its copied temporary, not a further copy.
void call_non_trivial_dtor(NonTrivialDtor &n) { arg_non_trivial_dtor(n); }

// CIR: cir.func {{.*}}@_Z21call_non_trivial_dtorR14NonTrivialDtor(
// CIR:   %[[TMP:.*]] = cir.alloca "agg.tmp0" align(4) : !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)>
// CIR-NEXT:   %[[TMP_VIEW:.*]] = cir.cast address_space %[[TMP]] : !cir.ptr<!rec_NonTrivialDtor, target_address_space(5)> -> !cir.ptr<!rec_NonTrivialDtor>
// CIR:   cir.copy %{{.*}} align(4) to %[[TMP_VIEW]] align(4) : !cir.ptr<!rec_NonTrivialDtor>
// CIR:   cir.call @_Z20arg_non_trivial_dtor14NonTrivialDtor(%[[TMP]]) {{.*}}: (!cir.ptr<!rec_NonTrivialDtor, target_address_space(5)> {{.*}}) -> ()
// CIR:   cir.call @_ZN14NonTrivialDtorD1Ev(%[[TMP_VIEW]])
// LLVM: define {{.*}}void @_Z21call_non_trivial_dtorR14NonTrivialDtor(
// LLVM:   %[[TMP:.*]] = alloca %struct.NonTrivialDtor, align 4, addrspace(5)
// LLVM:   %[[TMP_VIEW:.*]] = addrspacecast ptr addrspace(5) %[[TMP]] to ptr
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[TMP_VIEW]], ptr align 4 %{{.*}}, i64 4, i1 false)
// LLVM:   call void @_Z20arg_non_trivial_dtor14NonTrivialDtor(ptr addrspace(5) nofreeobj noundef align 4 dereferenceable(4) %{{.*}})
// LLVM:   call void @_ZN14NonTrivialDtorD1Ev(ptr {{.*}}%[[TMP_VIEW]])
