// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void b(void *__attribute__((pass_object_size(0))));
void e(void *__attribute__((pass_object_size(2))));

// CIR: cir.func private @b(!cir.ptr<!void> {llvm.noundef}, !u64i {llvm.noundef})

void test_constant() {
  int a;
  b(&a);
}

// CIR: cir.func {{.*}} @test_constant()
// CIR:   %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CIR:   %[[CAST:.*]] = cir.cast bitcast %[[ALLOCA]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:   %[[SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   cir.call @b(%[[CAST]], %[[SIZE]]) : (!cir.ptr<!void> {{.*}}, !u64i {{.*}}) -> ()

// CIR: cir.func private @e(!cir.ptr<!void> {llvm.noundef}, !u64i {llvm.noundef})

// LLVM: declare void @b(ptr noundef, i64 noundef)

// LLVM: define dso_local void @test_constant()
// LLVM:   %[[ALLOCA:.*]] = alloca i32
// LLVM:   call void @b(ptr noundef %[[ALLOCA]], i64 noundef 4)

// OGCG: define dso_local void @test_constant()
// OGCG:   %[[A:.*]] = alloca i32
// OGCG:   call void @b(ptr noundef %[[A]], i64 noundef 4)

// OGCG: declare void @b(ptr noundef, i64 noundef)

void test_vla(int n) {
  int d[n];
  b(d);
  e(d);
}

// CIR: cir.func {{.*}} @test_vla
// CIR:   %[[VLA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %{{.*}} : !u64i, ["d"] {alignment = 16 : i64}
// CIR:   %[[CAST1:.*]] = cir.cast bitcast %[[VLA]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:   %[[SIZE1:.*]] = cir.objsize max nullunknown %[[CAST1]] : !cir.ptr<!void> -> !u64i
// CIR:   cir.call @b(%[[CAST1]], %[[SIZE1]]) : (!cir.ptr<!void> {{.*}}, !u64i {{.*}}) -> ()
// CIR:   %[[CAST2:.*]] = cir.cast bitcast %[[VLA]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR:   %[[SIZE2:.*]] = cir.objsize min nullunknown %[[CAST2]] : !cir.ptr<!void> -> !u64i
// CIR:   cir.call @e(%[[CAST2]], %[[SIZE2]]) : (!cir.ptr<!void> {{.*}}, !u64i {{.*}}) -> ()

// LLVM: define dso_local void @test_vla(i32 noundef %{{.*}})
// LLVM:   %[[VLA:.*]] = alloca i32, i64 %{{.*}}, align 16
// LLVM:   %[[SIZE1:.*]] = call i64 @llvm.objectsize.i64.p0(ptr %[[VLA]], i1 false, i1 true, i1 false)
// LLVM:   call void @b(ptr noundef %[[VLA]], i64 noundef %[[SIZE1]])
// LLVM:   %[[SIZE2:.*]] = call i64 @llvm.objectsize.i64.p0(ptr %[[VLA]], i1 true, i1 true, i1 false)
// LLVM:   call void @e(ptr noundef %[[VLA]], i64 noundef %[[SIZE2]])

// OGCG: define dso_local void @test_vla(i32 noundef %{{.*}})
// OGCG:   %[[VLA:.*]] = alloca i32, i64 %{{.*}}, align 16
// OGCG:   %[[SIZE1:.*]] = call i64 @llvm.objectsize.i64.p0(ptr %[[VLA]], i1 false, i1 true, i1 false)
// OGCG:   call void @b(ptr noundef %[[VLA]], i64 noundef %[[SIZE1]])
// OGCG:   %[[SIZE2:.*]] = call i64 @llvm.objectsize.i64.p0(ptr %[[VLA]], i1 true, i1 true, i1 false)
// OGCG:   call void @e(ptr noundef %[[VLA]], i64 noundef %[[SIZE2]])
