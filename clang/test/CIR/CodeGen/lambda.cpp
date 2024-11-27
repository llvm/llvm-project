// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir  -emit-llvm -o - %s \
// RUN: | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void fn() {
  auto a = [](){};
  a();
}

//      CHECK: !ty_anon2E0_ = !cir.struct<class "anon.0" {!u8i}>
//  CHECK-DAG: module

//      CHECK: cir.func lambda internal private @_ZZ2fnvENK3$_0clEv{{.*}}) extra

//      CHECK:   cir.func @_Z2fnv()
// CHECK-NEXT:     %0 = cir.alloca !ty_anon2E0_, !cir.ptr<!ty_anon2E0_>, ["a"]
//      CHECK:   cir.call @_ZZ2fnvENK3$_0clEv

// LLVM: {{.*}}void @"_ZZ2fnvENK3$_0clEv"(ptr [[THIS:%.*]])
// FIXME: argument attributes should be emmitted, and lambda's alignment
// COM: LLVM: {{.*}} @"_ZZ2fnvENK3$_0clEv"(ptr noundef nonnull align 1 dereferenceable(1) [[THIS:%.*]]){{%.*}} align 2 {
// LLVM: [[THIS_ADDR:%.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// LLVM: [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// LLVM: ret void

// LLVM-LABEL: _Z2fnv
// LLVM:  [[a:%.*]] = alloca %class.anon.0, i64 1, align 1
// FIXME: parameter attributes should be emitted
// LLVM:  call void @"_ZZ2fnvENK3$_0clEv"(ptr [[a]])
// COM: LLVM:  call void @"_ZZ2fnvENK3$_0clEv"(ptr noundef nonnull align 1 dereferenceable(1) [[a]])
// LLVM:  ret void

void l0() {
  int i;
  auto a = [&](){ i = i + 1; };
  a();
}

// CHECK: cir.func lambda internal private @_ZZ2l0vENK3$_0clEv({{.*}}) extra

// CHECK: %0 = cir.alloca !cir.ptr<!ty_anon2E2_>, !cir.ptr<!cir.ptr<!ty_anon2E2_>>, ["this", init] {alignment = 8 : i64}
// CHECK: cir.store %arg0, %0 : !cir.ptr<!ty_anon2E2_>, !cir.ptr<!cir.ptr<!ty_anon2E2_>>
// CHECK: %1 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_anon2E2_>>, !cir.ptr<!ty_anon2E2_>
// CHECK: %2 = cir.get_member %1[0] {name = "i"} : !cir.ptr<!ty_anon2E2_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %3 = cir.load %2 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %4 = cir.load %3 : !cir.ptr<!s32i>, !s32i
// CHECK: %5 = cir.const #cir.int<1> : !s32i
// CHECK: %6 = cir.binop(add, %4, %5) nsw : !s32i
// CHECK: %7 = cir.get_member %1[0] {name = "i"} : !cir.ptr<!ty_anon2E2_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %8 = cir.load %7 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: cir.store %6, %8 : !s32i, !cir.ptr<!s32i>

// CHECK: cir.func @_Z2l0v()

// LLVM: {{.* }}void @"_ZZ2l0vENK3$_0clEv"(ptr [[THIS:%.*]])
// LLVM: [[THIS_ADDR:%.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// LLVM: [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// LLVM: [[I:%.*]] = getelementptr %class.anon.2, ptr [[THIS1]], i32 0, i32 0
// FIXME: getelementptr argument attributes should be emitted
// COM: LLVM: [[I:%.*]] = getelementptr inbounds nuw %class.anon.0, ptr [[THIS1]], i32 0, i32 0
// LLVM: [[TMP0:%.*]] = load ptr, ptr [[I]], align 8
// LLVM: [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// LLVM: [[ADD:%.*]] = add nsw i32 [[TMP1]], 1
// LLVM: [[I:%.*]] = getelementptr %class.anon.2, ptr [[THIS1]], i32 0, i32
// COM: LLVM: [[I:%.*]] = getelementptr inbounds nuw %class.anon.0, ptr [[THIS1]], i32 0, i32 0
// LLVM: [[TMP4:%.*]] = load ptr, ptr [[I]], align 8
// LLVM: store i32 [[ADD]], ptr [[TMP4]], align 4
// LLVM: ret void

// LLVM-LABEL: _Z2l0v
// LLVM:  [[i:%.*]] = alloca i32, i64 1, align 4
// LLVM:  [[a:%.*]] = alloca %class.anon.2, i64 1, align 8
// FIXME: getelementptr argument attributes should be emitted
// COM: LLVM:  [[TMP0:%.*]] = getelementptr inbounds %class.anon.2, ptr [[a]], i32 0, i32 0
// LLVM:  [[TMP0:%.*]] = getelementptr %class.anon.2, ptr [[a]], i32 0, i32 0
// LLVM:  store ptr [[i]], ptr [[TMP0]], align 8
// FIXME: parameter attributes should be emitted
// COM: LLVM:  call void @"_ZZ2l0vENK3$_0clEv"(ptr noundef nonnull align 1 dereferenceable(1) [[a]])
// LLVM:  call void @"_ZZ2l0vENK3$_0clEv"(ptr [[a]])
// LLVM:  ret void

auto g() {
  int i = 12;
  return [&] {
    i += 100;
    return i;
  };
}

// CHECK: cir.func @_Z1gv() -> !ty_anon2E3_
// CHECK: %0 = cir.alloca !ty_anon2E3_, !cir.ptr<!ty_anon2E3_>, ["__retval"] {alignment = 8 : i64}
// CHECK: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK: %2 = cir.const #cir.int<12> : !s32i
// CHECK: cir.store %2, %1 : !s32i, !cir.ptr<!s32i>
// CHECK: %3 = cir.get_member %0[0] {name = "i"} : !cir.ptr<!ty_anon2E3_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK: cir.store %1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %4 = cir.load %0 : !cir.ptr<!ty_anon2E3_>, !ty_anon2E3_
// CHECK: cir.return %4 : !ty_anon2E3_

// LLVM-LABEL: @_Z1gv()
// LLVM: [[retval:%.*]] = alloca %class.anon.3, i64 1, align 8
// LLVM: [[i:%.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 12, ptr [[i]], align 4
// LLVM: [[i_addr:%.*]] = getelementptr %class.anon.3, ptr [[retval]], i32 0, i32 0
// LLVM: store ptr [[i]], ptr [[i_addr]], align 8
// LLVM: [[tmp:%.*]] = load %class.anon.3, ptr [[retval]], align 8
// LLVM: ret %class.anon.3 [[tmp]]

auto g2() {
  int i = 12;
  auto lam = [&] {
    i += 100;
    return i;
  };
  return lam;
}

// Should be same as above because of NRVO
// CHECK: cir.func @_Z2g2v() -> !ty_anon2E4_
// CHECK-NEXT: %0 = cir.alloca !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>, ["__retval", init] {alignment = 8 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.const #cir.int<12> : !s32i
// CHECK-NEXT: cir.store %2, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %3 = cir.get_member %0[0] {name = "i"} : !cir.ptr<!ty_anon2E4_> -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.store %1, %3 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %4 = cir.load %0 : !cir.ptr<!ty_anon2E4_>, !ty_anon2E4_
// CHECK-NEXT: cir.return %4 : !ty_anon2E4_

// LLVM-LABEL: @_Z2g2v()
// LLVM: [[retval:%.*]] = alloca %class.anon.4, i64 1, align 8
// LLVM: [[i:%.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 12, ptr [[i]], align 4
// LLVM: [[i_addr:%.*]] = getelementptr %class.anon.4, ptr [[retval]], i32 0, i32 0
// LLVM: store ptr [[i]], ptr [[i_addr]], align 8
// LLVM: [[tmp:%.*]] = load %class.anon.4, ptr [[retval]], align 8
// LLVM: ret %class.anon.4 [[tmp]]

int f() {
  return g2()();
}

//      CHECK: cir.func @_Z1fv() -> !s32i
// CHECK-NEXT:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %2 = cir.alloca !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK-NEXT:     %3 = cir.call @_Z2g2v() : () -> !ty_anon2E4_
// CHECK-NEXT:     cir.store %3, %2 : !ty_anon2E4_, !cir.ptr<!ty_anon2E4_>
// CHECK-NEXT:     %4 = cir.call @_ZZ2g2vENK3$_0clEv(%2) : (!cir.ptr<!ty_anon2E4_>) -> !s32i
// CHECK-NEXT:     cir.store %4, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.return %1 : !s32i
// CHECK-NEXT: }

// LLVM: {{.*}}i32 @"_ZZ2g2vENK3$_0clEv"(ptr [[THIS:%.*]])
// LLVM: [[THIS_ADDR:%.*]] = alloca ptr, i64 1, align 8
// LLVM: [[I_SAVE:%.*]] = alloca i32, i64 1, align 4
// LLVM: store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// LLVM: [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// LLVM: [[I:%.*]] = getelementptr %class.anon.4, ptr [[THIS1]], i32 0, i32 0
// LLVM: [[TMP0:%.*]] = load ptr, ptr [[I]], align 8
// LLVM: [[TMP1:%.*]] = load i32, ptr [[TMP0]], align 4
// LLVM: [[ADD:%.*]] = add nsw i32 [[TMP1]], 100
// LLVM: [[I:%.*]] = getelementptr %class.anon.4, ptr [[THIS1]], i32 0, i32 0
// LLVM: [[TMP4:%.*]] = load ptr, ptr [[I]], align 8
// LLVM: [[TMP5:%.*]] = load i32, ptr [[TMP4]], align 4
// LLVM: store i32 [[TMP5]], ptr [[I_SAVE]], align 4
// LLVM: [[TMP6:%.*]] = load i32, ptr [[I_SAVE]], align 4
// LLVM: ret i32 [[TMP6]]

// LLVM-LABEL: _Z1fv
// LLVM: [[ref_tmp0:%.*]] = alloca %class.anon.4, i64 1, align 8
// LLVM: [[ret_val:%.*]] = alloca i32, i64 1, align 4
// LLVM: br label %[[scope_bb:[0-9]+]]
// LLVM: [[scope_bb]]:
// LLVM: [[tmp0:%.*]] = call %class.anon.4 @_Z2g2v()
// LLVM: store %class.anon.4 [[tmp0]], ptr [[ref_tmp0]], align 8
// LLVM: [[tmp1:%.*]] = call i32 @"_ZZ2g2vENK3$_0clEv"(ptr [[ref_tmp0]])
// LLVM: store i32 [[tmp1]], ptr [[ret_val]], align 4
// LLVM: br label %[[ret_bb:[0-9]+]]
// LLVM: [[ret_bb]]:
// LLVM: [[tmp2:%.*]] = load i32, ptr [[ret_val]], align 4
// LLVM: ret i32 [[tmp2]]

int g3() {
  auto* fn = +[](int const& i) -> int { return i; };
  auto task = fn(3);
  return task;
}

// lambda operator()
// CHECK: cir.func lambda internal private @_ZZ2g3vENK3$_0clERKi{{.*}}!s32i extra

// lambda __invoke()
// CHECK:   cir.func internal private @_ZZ2g3vEN3$_08__invokeERKi

// lambda operator int (*)(int const&)()
// CHECK:   cir.func internal private @_ZZ2g3vENK3$_0cvPFiRKiEEv

// CHECK: cir.func @_Z2g3v() -> !s32i
// CHECK:     %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>, ["fn", init] {alignment = 8 : i64}
// CHECK:     %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["task", init] {alignment = 4 : i64}

// 1. Use `operator int (*)(int const&)()` to retrieve the fnptr to `__invoke()`.
// CHECK:     %3 = cir.scope {
// CHECK:       %7 = cir.alloca !ty_anon2E5_, !cir.ptr<!ty_anon2E5_>, ["ref.tmp0"] {alignment = 1 : i64}
// CHECK:       %8 = cir.call @_ZZ2g3vENK3$_0cvPFiRKiEEv(%7) : (!cir.ptr<!ty_anon2E5_>) -> !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       %9 = cir.unary(plus, %8) : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       cir.yield %9 : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:     }

// 2. Load ptr to `__invoke()`.
// CHECK:     cir.store %3, %1 : !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>
// CHECK:     %4 = cir.scope {
// CHECK:       %7 = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp1", init] {alignment = 4 : i64}
// CHECK:       %8 = cir.load %1 : !cir.ptr<!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>
// CHECK:       %9 = cir.const #cir.int<3> : !s32i
// CHECK:       cir.store %9, %7 : !s32i, !cir.ptr<!s32i>

// 3. Call `__invoke()`, which effectively executes `operator()`.
// CHECK:       %10 = cir.call %8(%7) : (!cir.ptr<!cir.func<!s32i (!cir.ptr<!s32i>)>>, !cir.ptr<!s32i>) -> !s32i
// CHECK:       cir.yield %10 : !s32i
// CHECK:     }

// CHECK:   }

// lambda operator()
// FIXME: argument attributes should be emitted
// COM: LLVM: define internal noundef i32 @"_ZZ2g3vENK3$_0clERKi"(ptr noundef nonnull align 1 dereferenceable(1) {{%.*}}, ptr noundef nonnull align 4 dereferenceable(4){{%.*}}) #0 align 2
// LLVM: {{.*}}i32 @"_ZZ2g3vENK3$_0clERKi"(ptr {{%.*}}, ptr {{%.*}})

// lambda __invoke()
// LLVM: {{.*}}i32 @"_ZZ2g3vEN3$_08__invokeERKi"(ptr [[i:%.*]])
// LLVM: [[i_addr:%.*]] = alloca ptr, i64 1, align 8
// LLVM: [[ret_val:%.*]] = alloca i32, i64 1, align 4
// LLVM: [[unused_capture:%.*]] = alloca %class.anon.5, i64 1, align 1
// LLVM: store ptr [[i]], ptr [[i_addr]], align 8
// LLVM: [[TMP0:%.*]] = load ptr, ptr [[i_addr]], align 8
// FIXME: call and argument attributes should be emitted
// COM: LLVM: [[CALL:%.*]] =  call noundef i32 @"_ZZ2g3vENK3$_0clERKi"(ptr noundef nonnull align 1 dereferenceable(1) [[unused_capture]], ptr noundef nonnull align 4 dereferenceable(4) [[TMP0]])
// LLVM: [[CALL:%.*]] = call i32 @"_ZZ2g3vENK3$_0clERKi"(ptr [[unused_capture]], ptr [[TMP0]])
// LLVM: store i32 [[CALL]], ptr [[ret_val]], align 4
// LLVM: %[[ret:.*]] = load i32, ptr [[ret_val]], align 4
// LLVM: ret i32 %[[ret]]

// lambda operator int (*)(int const&)()
// LLVM-LABEL: @"_ZZ2g3vENK3$_0cvPFiRKiEEv"
// LLVM:  store ptr @"_ZZ2g3vEN3$_08__invokeERKi", ptr [[ret_val:%.*]], align 8
// LLVM:  [[TMP0:%.*]] = load ptr, ptr [[ret_val]], align 8
// LLVM:  ret ptr [[TMP0]]

// LLVM-LABEL: _Z2g3v
// LLVM-DAG: [[ref_tmp0:%.*]] = alloca %class.anon.5, i64 1, align 1
// LLVM-DAG: [[ref_tmp1:%.*]] = alloca i32, i64 1, align 4
// LLVM-DAG: [[ret_val:%.*]] = alloca i32, i64 1, align 4
// LLVM-DAG: [[fn_ptr:%.*]] = alloca ptr, i64 1, align 8
// LLVM-DAG: [[task:%.*]] = alloca i32, i64 1, align 4
// LLVM: br label %[[scope0_bb:[0-9]+]]

// LLVM: [[scope0_bb]]: {{.*}}; preds = %0
// LLVM: [[call:%.*]] = call ptr @"_ZZ2g3vENK3$_0cvPFiRKiEEv"(ptr [[ref_tmp0]])
// LLVM: br label %[[scope1_before:[0-9]+]]

// LLVM: [[scope1_before]]: {{.*}}; preds = %[[scope0_bb]]
// LLVM: [[tmp0:%.*]] = phi ptr [ [[call]], %[[scope0_bb]] ]
// LLVM: br label %[[scope1_bb:[0-9]+]]

// LLVM: [[scope1_bb]]: {{.*}}; preds = %[[scope1_before]]
// LLVM: [[fn:%.*]] = load ptr, ptr [[fn_ptr]], align 8
// LLVM: store i32 3, ptr [[ref_tmp1]], align 4
// LLVM: [[call1:%.*]] = call i32 [[fn]](ptr [[ref_tmp1]])
// LLVM: br label %[[ret_bb:[0-9]+]]

// LLVM: [[ret_bb]]: {{.*}}; preds = %[[scope1_bb]]
// LLVM: [[tmp1:%.*]] = phi i32 [ [[call1]], %[[scope1_bb]] ]
// LLVM: store i32 [[tmp1]], ptr [[task]], align 4
// LLVM: [[tmp2:%.*]] = load i32, ptr [[task]], align 4
// LLVM: store i32 [[tmp2]], ptr [[ret_val]], align 4
// LLVM: [[tmp3:%.*]] = load i32, ptr [[ret_val]], align 4
// LLVM: ret i32 [[tmp3]]
