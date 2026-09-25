// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++03 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++03 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++03 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG

struct Struct {
  int member;
  Struct(int);
};

extern "C" Struct getStruct(int i) { return i; }

// CIR: cir.func {{.*}} @getStruct(%{{[^,)]+}}: !s32i {{.*}}) -> !s32i
// LLVM: define dso_local i32 @getStruct(i32 noundef %{{[^,)]+}})

extern "C" void use() {
  int g = getStruct(0).member;

  // CIR-LABEL: @use()
  // CIR: %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!s32i>
  // CIR: %[[G_ALLOCA:.*]] = cir.alloca "g" {{.*}} init : !cir.ptr<!s32i>
  // CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca {{.*}} : !cir.ptr<!rec_Struct>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[GET_STRUCT_CALL:.*]] = cir.call @getStruct(%[[ZERO]])
  // CIR: cir.store{{.*}} %[[GET_STRUCT_CALL]], %[[COERCE]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[COERCE_REC:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!s32i> -> !cir.ptr<!rec_Struct>
  // CIR: %[[STRUCT_VAL:.*]] = cir.load %[[COERCE_REC]] : !cir.ptr<!rec_Struct>, !rec_Struct
  // CIR: cir.store{{.*}} %[[STRUCT_VAL]], %[[TEMP_ALLOCA]]
  // CIR: %[[GET_MEMBER:.*]] = cir.get_member %[[TEMP_ALLOCA]][0] {name = "member"}
  // CIR: %[[LOAD_MEM:.*]] = cir.load{{.*}}%[[GET_MEMBER]]
  // CIR: cir.store{{.*}} %[[LOAD_MEM]], %[[G_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // LLVM-LABEL: @use()
  // LLVMCIR: %[[COERCE:.*]] = alloca i32
  // LLVMCIR: %[[G_ALLOCA:.*]] = alloca i32
  // LLVMCIR: %[[TEMP_ALLOCA:.*]] = alloca %struct.Struct
  // LLVMCIR: %[[GET_STRUCT_CALL:.*]] = call i32 @getStruct(i32 noundef 0)
  // LLVMCIR: store i32 %[[GET_STRUCT_CALL]], ptr %[[COERCE]]
  // LLVMCIR: %[[STRUCT_VAL:.*]] = load %struct.Struct, ptr %[[COERCE]]
  // LLVMCIR: store %struct.Struct %[[STRUCT_VAL]], ptr %[[TEMP_ALLOCA]]
  //
  // OGCG: %[[G_ALLOCA:.*]] = alloca i32
  // OGCG: %[[TEMP_ALLOCA:.*]] = alloca %struct.Struct
  // OGCG: %[[GET_STRUCT_CALL:.*]] = call i32 @getStruct(i32 noundef 0)
  // OGCG: %[[COERCE_DIVE:.*]] = getelementptr{{.*}}%struct.Struct, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
  // OGCG: store i32 %[[GET_STRUCT_CALL]], ptr %[[COERCE_DIVE]]
  //
  // LLVM: %[[GET_MEMBER:.*]] = getelementptr {{.*}}%struct.Struct, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
  // LLVM: %[[LOAD_MEM:.*]] = load i32, ptr %[[GET_MEMBER]]
  // LLVM: store i32 %[[LOAD_MEM]], ptr %[[G_ALLOCA]]
}

