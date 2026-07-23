// RUN: %clang_cc1 -std=c++03 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++03 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++03 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct Pt {
  Pt(int a, int b);
  int v;
};

int tempObj(int i) { return Pt(i, i).v; }

// CIR-LABEL: cir.func {{.*}}@_Z7tempObji(%arg0: !s32i
// CIR:         %[[I:.*]] = cir.alloca "i"
// CIR:         %[[RET:.*]] = cir.alloca "__retval"
// CIR:         %[[TMP:.*]] = cir.alloca "tmp"
// CIR:         cir.store %arg0, %[[I]]
// CIR:         %[[A:.*]] = cir.load align(4) %[[I]]
// CIR:         %[[B:.*]] = cir.load align(4) %[[I]]
// CIR:         cir.call @_ZN2PtC1Eii(%[[TMP]], %[[A]], %[[B]])
// CIR:         %[[V:.*]] = cir.get_member %[[TMP]][0] {name = "v"} : !cir.ptr<!rec_Pt> -> !cir.ptr<!s32i>
// CIR:         %[[VAL:.*]] = cir.load align(4) %[[V]]
// CIR:         cir.store %[[VAL]], %[[RET]]
// CIR:         %[[RES:.*]] = cir.load %[[RET]]
// CIR:         cir.return %[[RES]]

// LLVM-LABEL: define {{.*}}i32 @_Z7tempObji
// LLVM:         %[[I_ADDR:.*]] = alloca i32
// LLVM:         %[[TMP:.*]] = alloca %struct.Pt
// LLVM:         store i32 %{{.*}}, ptr %[[I_ADDR]]
// LLVM:         %[[A:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:         %[[B:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:         call void @_ZN2PtC1Eii(ptr {{.*}} %[[TMP]], i32 {{.*}} %[[A]], i32 {{.*}} %[[B]])
// LLVM:         %[[V:.*]] = getelementptr inbounds nuw %struct.Pt, ptr %[[TMP]], i32 0, i32 0
// LLVM:         %[[VAL:.*]] = load i32, ptr %[[V]]

struct Conv {
  Conv(int a);
  int y;
};

int construct(int i) { return Conv(i).y; }

// CIR-LABEL: cir.func {{.*}}@_Z9constructi(%arg0: !s32i
// CIR:         %[[I:.*]] = cir.alloca "i"
// CIR:         %[[RET:.*]] = cir.alloca "__retval"
// CIR:         %[[TMP:.*]] = cir.alloca "tmp"
// CIR:         cir.store %arg0, %[[I]]
// CIR:         %[[A:.*]] = cir.load align(4) %[[I]]
// CIR:         cir.call @_ZN4ConvC1Ei(%[[TMP]], %[[A]])
// CIR:         %[[Y:.*]] = cir.get_member %[[TMP]][0] {name = "y"} : !cir.ptr<!rec_Conv> -> !cir.ptr<!s32i>
// CIR:         %[[VAL:.*]] = cir.load align(4) %[[Y]]
// CIR:         cir.store %[[VAL]], %[[RET]]
// CIR:         %[[RES:.*]] = cir.load %[[RET]]
// CIR:         cir.return %[[RES]]

// LLVM-LABEL: define {{.*}}i32 @_Z9constructi
// LLVM:         %[[I_ADDR:.*]] = alloca i32
// LLVM:         %[[TMP:.*]] = alloca %struct.Conv
// LLVM:         store i32 %{{.*}}, ptr %[[I_ADDR]]
// LLVM:         %[[A:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:         call void @_ZN4ConvC1Ei(ptr {{.*}} %[[TMP]], i32 {{.*}} %[[A]])
// LLVM:         %[[Y:.*]] = getelementptr inbounds nuw %struct.Conv, ptr %[[TMP]], i32 0, i32 0
// LLVM:         %[[VAL:.*]] = load i32, ptr %[[Y]]
