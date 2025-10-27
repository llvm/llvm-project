// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Point {
  int x, y;
};

struct Line {
  struct Point start;
  struct Point end;
};

// AggExprEmitter::VisitMemberExpr
void test_member_in_array(void) {
  struct Line line = {{1, 2}, {3, 4}};
  struct Point arr[1] = {line.start};
}

// CIR-LABEL: cir.func{{.*}} @test_member_in_array
// CIR:   %[[LINE:.*]] = cir.alloca !rec_Line{{.*}}, ["line", init]
// CIR:   %[[ARR:.*]] = cir.alloca !cir.array<!rec_Point x 1>{{.*}}, ["arr", init]
// CIR:   %[[MEMBER:.*]] = cir.get_member %[[LINE]][0] {name = "start"}
// CIR:   cir.copy

// LLVM-LABEL: define{{.*}} @test_member_in_array
// LLVM:   %[[LINE:.*]] = alloca %struct.Line
// LLVM:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// LLVM:   %[[MEMBER:.*]] = getelementptr{{.*}}%struct.Line{{.*}}%[[LINE]]{{.*}}i32 0, i32 0
// LLVM:   call void @llvm.memcpy

// OGCG-LABEL: define{{.*}} @test_member_in_array
// OGCG:   %[[LINE:.*]] = alloca %struct.Line
// OGCG:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// OGCG:   %[[MEMBER:.*]] = getelementptr{{.*}}%struct.Line{{.*}}%[[LINE]]{{.*}}i32 0, i32 0
// OGCG:   call void @llvm.memcpy

// AggExprEmitter::VisitMemberExpr
void test_member_arrow_in_array(void) {
  struct Line *line_ptr;
  struct Point arr[1] = {line_ptr->start};
}

// CIR-LABEL: cir.func{{.*}} @test_member_arrow_in_array
// CIR:   %[[PTR:.*]] = cir.alloca !cir.ptr<!rec_Line>{{.*}}, ["line_ptr"]
// CIR:   %[[ARR:.*]] = cir.alloca !cir.array<!rec_Point x 1>{{.*}}, ["arr", init]
// CIR:   %[[LOADED:.*]] = cir.load{{.*}}%[[PTR]]
// CIR:   %[[MEMBER:.*]] = cir.get_member %[[LOADED]][0] {name = "start"}
// CIR:   cir.copy

// LLVM-LABEL: define{{.*}} @test_member_arrow_in_array
// LLVM:   %[[PTR:.*]] = alloca ptr
// LLVM:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// LLVM:   %[[LOADED:.*]] = load ptr{{.*}}%[[PTR]]
// LLVM:   %[[MEMBER:.*]] = getelementptr{{.*}}%struct.Line{{.*}}%[[LOADED]]{{.*}}i32 0, i32 0
// LLVM:   call void @llvm.memcpy

// OGCG-LABEL: define{{.*}} @test_member_arrow_in_array
// OGCG:   %[[PTR:.*]] = alloca ptr
// OGCG:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// OGCG:   %[[LOADED:.*]] = load ptr{{.*}}%[[PTR]]
// OGCG:   %[[MEMBER:.*]] = getelementptr{{.*}}%struct.Line{{.*}}%[[LOADED]]{{.*}}i32 0, i32 0
// OGCG:   call void @llvm.memcpy

// AggExprEmitter::VisitUnaryDeref
void test_deref_in_array(void) {
  struct Point *ptr;
  struct Point arr[1] = {*ptr};
}

// CIR-LABEL: cir.func{{.*}} @test_deref_in_array
// CIR:   %[[PTR:.*]] = cir.alloca !cir.ptr<!rec_Point>{{.*}}, ["ptr"]
// CIR:   %[[ARR:.*]] = cir.alloca !cir.array<!rec_Point x 1>{{.*}}, ["arr", init]
// CIR:   %[[LOADED:.*]] = cir.load{{.*}}%[[PTR]]
// CIR:   cir.copy

// LLVM-LABEL: define{{.*}} @test_deref_in_array
// LLVM:   %[[PTR:.*]] = alloca ptr
// LLVM:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// LLVM:   %[[LOADED:.*]] = load ptr{{.*}}%[[PTR]]
// LLVM:   call void @llvm.memcpy

// OGCG-LABEL: define{{.*}} @test_deref_in_array
// OGCG:   %[[PTR:.*]] = alloca ptr
// OGCG:   %[[ARR:.*]] = alloca [1 x %struct.Point]
// OGCG:   %[[LOADED:.*]] = load ptr{{.*}}%[[PTR]]
// OGCG:   call void @llvm.memcpy

// AggExprEmitter::VisitStringLiteral
void test_string_array_in_array(void) {
    char matrix[2][6] = {"hello", "world"};
}
  
// CIR-LABEL: cir.func{{.*}} @test_string_array_in_array
// CIR:   cir.alloca !cir.array<!cir.array<!s8i x 6> x 2>, {{.*}}, ["matrix", init]
// CIR:   cir.get_global
// CIR:   cir.copy
// CIR:   cir.get_global
// CIR:   cir.copy

// LLVM-LABEL: define{{.*}} @test_string_array_in_array
// LLVM:   alloca [2 x [6 x i8]]
// LLVM:   call void @llvm.memcpy
// LLVM:   call void @llvm.memcpy

// OGCG-LABEL: define{{.*}} @test_string_array_in_array
// OGCG:   alloca [2 x [6 x i8]]
// OGCG:   call void @llvm.memcpy{{.*}}@__const.test_string_array_in_array
