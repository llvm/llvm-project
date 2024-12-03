// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct HasScalarArrayMember {
  int arr[2][2];
  HasScalarArrayMember(const HasScalarArrayMember &);
};

// CIR-LABEL: cir.func @_ZN20HasScalarArrayMemberC2ERKS_(
// CIR-NEXT:    %[[#THIS:]] = cir.alloca !cir.ptr<!ty_HasScalarArrayMember>
// CIR-NEXT:    %[[#OTHER:]] = cir.alloca !cir.ptr<!ty_HasScalarArrayMember>
// CIR-NEXT:    cir.store %arg0, %[[#THIS]]
// CIR-NEXT:    cir.store %arg1, %[[#OTHER]]
// CIR-NEXT:    %[[#THIS_LOAD:]] = cir.load %[[#THIS]]
// CIR-NEXT:    %[[#THIS_ARR:]] = cir.get_member %[[#THIS_LOAD]][0] {name = "arr"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CIR-NEXT:    %[[#OTHER_ARR:]] = cir.get_member %[[#OTHER_LOAD]][0] {name = "arr"}
// CIR-NEXT:    cir.copy %[[#OTHER_ARR]] to %[[#THIS_ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 2> x 2>>
// CIR-NEXT:    cir.return

// LLVM-LABEL: define {{.*}} @_ZN20HasScalarArrayMemberC2ERKS_(
// LLVM-SAME:      ptr %[[#ARG0:]], ptr %[[#ARG1:]])
// LLVM-NEXT:    %[[#THIS:]] = alloca ptr
// LLVM-NEXT:    %[[#OTHER:]] = alloca ptr
// LLVM-NEXT:    store ptr %[[#ARG0]], ptr %[[#THIS]]
// LLVM-NEXT:    store ptr %[[#ARG1]], ptr %[[#OTHER]]
// LLVM-NEXT:    %[[#THIS_LOAD:]] = load ptr, ptr %[[#THIS]]
// LLVM-NEXT:    %[[#THIS_ARR:]] = getelementptr %struct.HasScalarArrayMember, ptr %[[#THIS_LOAD]], i32 0, i32 0
// LLVM-NEXT:    %[[#OTHER_LOAD:]] = load ptr, ptr %[[#OTHER]]
// LLVM-NEXT:    %[[#OTHER_ARR:]] = getelementptr %struct.HasScalarArrayMember, ptr %[[#OTHER_LOAD]], i32 0, i32 0
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr %[[#THIS_ARR]], ptr %[[#OTHER_ARR]], i32 16, i1 false)
// LLVM-NEXT:    ret void
HasScalarArrayMember::HasScalarArrayMember(const HasScalarArrayMember &) = default;

struct Trivial { int *i; };
struct ManyMembers {
  int i;
  int j;
  Trivial k;
  int l[1];
  int m[2];
  Trivial n;
  int &o;
  int *p;
};

// CIR-LABEL: cir.func linkonce_odr @_ZN11ManyMembersC2ERKS_(
// CIR:         %[[#THIS_LOAD:]] = cir.load %[[#]]
// CIR-NEXT:    %[[#THIS_I:]] = cir.get_member %[[#THIS_LOAD]][0] {name = "i"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER:]]
// CIR-NEXT:    %[[#OTHER_I:]] = cir.get_member %[[#OTHER_LOAD]][0] {name = "i"}
// CIR-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<8>
// CIR-NEXT:    %[[#THIS_I_CAST:]] = cir.cast(bitcast, %[[#THIS_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR-NEXT:    %[[#OTHER_I_CAST:]] = cir.cast(bitcast, %[[#OTHER_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_I_CAST]] to %[[#THIS_I_CAST]]
// CIR-NEXT:    %[[#THIS_K:]] = cir.get_member %[[#THIS_LOAD]][2] {name = "k"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CIR-NEXT:    %[[#OTHER_K:]] = cir.get_member %[[#OTHER_LOAD]][2] {name = "k"}
// CIR-NEXT:    cir.call @_ZN7TrivialC1ERKS_(%[[#THIS_K]], %[[#OTHER_K]])
// CIR-NEXT:    %[[#THIS_L:]] = cir.get_member %[[#THIS_LOAD]][3] {name = "l"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CIR-NEXT:    %[[#OTHER_L:]] = cir.get_member %[[#OTHER_LOAD]][3] {name = "l"}
// CIR-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<12>
// CIR-NEXT:    %[[#THIS_L_CAST:]] = cir.cast(bitcast, %[[#THIS_L]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!void>
// CIR-NEXT:    %[[#OTHER_L_CAST:]] = cir.cast(bitcast, %[[#OTHER_L]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_L_CAST]] to %[[#THIS_L_CAST]]
// CIR-NEXT:    %[[#THIS_N:]] = cir.get_member %[[#THIS_LOAD]][5] {name = "n"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CIR-NEXT:    %[[#OTHER_N:]] = cir.get_member %[[#OTHER_LOAD]][5] {name = "n"}
// CIR-NEXT:    cir.call @_ZN7TrivialC1ERKS_(%[[#THIS_N]], %[[#OTHER_N]])
// CIR-NEXT:    %[[#THIS_O:]] = cir.get_member %[[#THIS_LOAD]][6] {name = "o"}
// CIR-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CIR-NEXT:    %[[#OTHER_O:]] = cir.get_member %[[#OTHER_LOAD]][6] {name = "o"}
// CIR-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<16>
// CIR-NEXT:    %[[#THIS_O_CAST:]] = cir.cast(bitcast, %[[#THIS_O]] : !cir.ptr<!cir.ptr<!s32i>>), !cir.ptr<!void>
// CIR-NEXT:    %[[#OTHER_O_CAST:]] = cir.cast(bitcast, %[[#OTHER_O]] : !cir.ptr<!cir.ptr<!s32i>>), !cir.ptr<!void>
// CIR-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_O_CAST]] to %[[#THIS_O_CAST]]
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

// CIR-LABEL: cir.func @_Z6doCopyR11ManyMembers(
// CIR:         cir.call @_ZN11ManyMembersC1ERKS_(
ManyMembers doCopy(ManyMembers &src) {
  return src;
}
