// RUN: %clang_cc1 -triple x86_64-linux-gnu  -gkey-instructions %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s

// Perennial question: should the inc be its own source atom or not
// (currently it is).

// FIXME: See do.c and while.c regarding cmp and cond br groups.

// The stores in the setup (stores to __RANGE1, __BEGIN1, __END1) are all
// marked as Key. Unclear whether that's desirable. Keep for now as that's
// least risky (at worst it introduces an unnecessary step while debugging,
// as opposed to potentially losing one we want).

// Check the conditional branch (and the condition) in FOR_COND and
// unconditional branch in FOR_BODY are Key Instructions.

struct Range {
    int *begin();
    int *end();
} r;

// CHECK-LABEL: define dso_local void @_Z1av(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] !dbg [[DBG10:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[__RANGE1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[__BEGIN1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[__END1:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr @r, ptr [[__RANGE1]], align 8, !dbg [[DBG14:![0-9]+]]
// CHECK-NEXT:    [[CALL:%.*]] = call noundef ptr @_ZN5Range5beginEv(ptr noundef nonnull align 1 dereferenceable(1) @r), !dbg [[DBG15:![0-9]+]]
// CHECK-NEXT:    store ptr [[CALL]], ptr [[__BEGIN1]], align 8, !dbg [[DBG16:![0-9]+]]
// CHECK-NEXT:    [[CALL1:%.*]] = call noundef ptr @_ZN5Range3endEv(ptr noundef nonnull align 1 dereferenceable(1) @r), !dbg [[DBG17:![0-9]+]]
// CHECK-NEXT:    store ptr [[CALL1]], ptr [[__END1]], align 8, !dbg [[DBG18:![0-9]+]]
// CHECK-NEXT:    br label %[[FOR_COND:.*]], !dbg [[DBG19:![0-9]+]]
// CHECK:       [[FOR_COND]]:
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[__BEGIN1]], align 8, !dbg [[DBG19]]
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[__END1]], align 8, !dbg [[DBG19]]
// CHECK-NEXT:    [[CMP:%.*]] = icmp ne ptr [[TMP0]], [[TMP1]], !dbg [[DBG20:![0-9]+]]
// CHECK-NEXT:    br i1 [[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END:.*]], !dbg [[DBG21:![0-9]+]]
// CHECK:       [[FOR_BODY]]:
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[__BEGIN1]], align 8, !dbg [[DBG19]]
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 4, !dbg [[DBG22:![0-9]+]]
// CHECK-NEXT:    store i32 [[TMP3]], ptr [[I]], align 4, !dbg [[DBG23:![0-9]+]]
// CHECK-NEXT:    br label %[[FOR_INC:.*]], !dbg [[DBG24:![0-9]+]]
// CHECK:       [[FOR_INC]]:
// CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[__BEGIN1]], align 8, !dbg [[DBG19]]
// CHECK-NEXT:    [[INCDEC_PTR:%.*]] = getelementptr inbounds nuw i32, ptr [[TMP4]], i32 1, !dbg [[DBG25:![0-9]+]]
// CHECK-NEXT:    store ptr [[INCDEC_PTR]], ptr [[__BEGIN1]], align 8, !dbg [[DBG26:![0-9]+]]
// CHECK-NEXT:    br label %[[FOR_COND]], !dbg [[DBG19]], !llvm.loop
// CHECK:       [[FOR_END]]:
// CHECK-NEXT:    ret void, !dbg [[DBG30:![0-9]+]]
//
void a() {
  for (int i: r)
    ;
}

// - Check the branch out of the body gets an atom group (and gets it correct
// if there's ctrl-flow in the body).
void b() {
  for (int i: r) {
    if (i)
      ;
// CHECK: entry:
// CHECK: if.end:
// CHECK-NEXT: br label %for.inc, !dbg [[b_br:!.*]]
  }
}

// - Check an immediate loop exit (`break` in this case) gets an atom group and
// - doesn't cause a crash.
void c() {
  for (int i: r)
// CHECK: entry:
// CHECK: for.body:
// CHECK: br label %for.end, !dbg [[c_br:!.*]]
    break;
}

//.
// CHECK: [[DBG14]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[DBG15]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[DBG16]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[DBG17]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
// CHECK: [[DBG18]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[DBG19]] = !DILocation({{.*}})
// CHECK: [[DBG20]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[DBG21]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[DBG22]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 2)
// CHECK: [[DBG23]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
// CHECK: [[DBG24]] = !DILocation({{.*}} atomGroup: 8, atomRank: 1)
// CHECK: [[DBG25]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 2)
// CHECK: [[DBG26]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)
// CHECK: [[DBG30]] = !DILocation({{.*}})
//
// CHECK: [[b_br]] = !DILocation({{.*}}, atomGroup: [[#]], atomRank: [[#]])
//
// CHECK: [[c_br]] = !DILocation({{.*}}, atomGroup: [[#]], atomRank: [[#]])
//.
