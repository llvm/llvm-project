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

extern "C" void use() {
  int g = getStruct(0).member;

  // CIR-LABEL: @use()
  // CIR: %[[G_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["g", init]
  // CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca !rec_Struct, !cir.ptr<!rec_Struct>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[GET_STRUCT_CALL:.*]] = cir.call @getStruct(%[[ZERO]])
  // CIR: cir.store{{.*}} %[[GET_STRUCT_CALL]], %[[TEMP_ALLOCA]]
  // CIR: %[[GET_MEMBER:.*]] = cir.get_member %[[TEMP_ALLOCA]][0] {name = "member"}
  // CIR: %[[LOAD_MEM:.*]] = cir.load{{.*}}%[[GET_MEMBER]]
  // CIR: cir.store{{.*}} %[[LOAD_MEM]], %[[G_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // LLVM-LABEL: @use()
  // LLVM: %[[G_ALLOCA:.*]] = alloca i32
  // LLVM: %[[TEMP_ALLOCA:.*]] = alloca %struct.Struct
  //
  // LLVMCIR: %[[GET_STRUCT_CALL:.*]] = call %struct.Struct @getStruct(i32 noundef 0)
  // LLVMCIR: store %struct.Struct %[[GET_STRUCT_CALL]], ptr %[[TEMP_ALLOCA]]
  // OGCG: %[[GET_STRUCT_CALL_BEFORE:.*]] = call i32 @getStruct(i32 noundef 0)
  // OGCG: %[[GET_STRUCT_CALL:.*]] = getelementptr{{.*}}%struct.Struct, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
  // OGCG: store i32 %[[GET_STRUCT_CALL_BEFORE]], ptr %[[GET_STRUCT_CALL]]
  //
  // LLVM: %[[GET_MEMBER:.*]] = getelementptr {{.*}}%struct.Struct, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
  // LLVM: %[[LOAD_MEM:.*]] = load i32, ptr %[[GET_MEMBER]]
  // LLVM: store i32 %[[LOAD_MEM]], ptr %[[G_ALLOCA]]
}

