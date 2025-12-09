// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Point {
  int x;
  int y;
  int z;
};

auto test1() -> int Point::* {
  return &Point::y;
}

// CIR: cir.func {{.*}} @_Z5test1v() -> !cir.data_member<!s32i in !rec_Point> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, ["__retval"]
// CIR:   %[[MEMBER:.*]] = cir.const #cir.data_member<1> : !cir.data_member<!s32i in !rec_Point>
// CIR:   cir.store %[[MEMBER]], %[[RETVAL]] : !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, !cir.data_member<!s32i in !rec_Point>
// CIR:   cir.return %[[RET]] : !cir.data_member<!s32i in !rec_Point>

// LLVM: define {{.*}} i64 @_Z5test1v()
// LLVM:   %[[RETVAL:.*]] = alloca i64
// LLVM:   store i64 4, ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z5test1v()
// OGCG:   ret i64 4
