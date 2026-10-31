// RUN: %clang_cc1 -std=c++20 -fexceptions -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefixes=CHECK,EH %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefixes=CHECK,NOEH %s

struct NoThrow {};
void *operator new(decltype(sizeof 0), NoThrow) noexcept;
void operator delete(void *, NoThrow) noexcept;

namespace GH175427 {
struct S {
  S(int);
};
[[noreturn]] void nr();

void WhileTrue() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754279WhileTrueEv()
  // CHECK:         call {{.*}}@_Znwm(i64 noundef 4)
  // EH-NEXT:       br label %while.cond
  // EH:          while.cond:
  // EH-NEXT:       br label %while.body
  // NOEH-NEXT:     br label %while.body
  // CHECK:       while.body:
  // EH-NEXT:       br label %while.cond
  // NOEH-NEXT:     br label %while.body
  new int(({
    while (true)
      ;
    1;
  }));
}

void ForFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754278ForFirstEv()
  // CHECK:         call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    br label %for.cond
  // CHECK:       for.cond:
  // CHECK-NEXT:    br label %for.cond
  new int(({
    for (;;)
      ;
    1;
  }));
}

void DoFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754277DoFirstEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    br label %do.body
  // CHECK:       do.body:
  // CHECK-NEXT:    br label %do.cond
  // CHECK:       do.cond:
  // CHECK-NEXT:    br i1 true, label %do.body, label %do.end
  // CHECK:       do.end:
  // CHECK:         store i32 {{.*}}, ptr [[CALL]]
  // CHECK-NEXT:    ret void
  new int(({
    do
      ;
    while (true);
    1;
  }));
}

void LabelFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH17542710LabelFirstEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    br label %L
  // CHECK:       L: ; preds = %entry
  // CHECK-NEXT:    store i32 1, ptr [[TMP:%[^,]+]]
  // CHECK-NEXT:    [[V:%.+]] = load i32, ptr [[TMP]]
  // CHECK-NEXT:    store i32 [[V]], ptr [[CALL]]
  // CHECK-NEXT:    ret void
  new int(({
  L:;
    1;
  }));
}

void WhileX(bool x) {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754276WhileXEb(
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    br label %while.cond
  // CHECK:       while.cond:
  // CHECK:         br i1 {{.*}}, label %while.body, label %while.end
  // CHECK:       while.body:
  // CHECK-NEXT:    br label %while.cond
  // CHECK:       while.end:
  // CHECK:         store i32 {{.*}}, ptr [[CALL]]
  // CHECK-NEXT:    ret void
  new int(({
    while (x)
      ;
    1;
  }));
}

void ArrayNew() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754278ArrayNewEv()
  // CHECK:         call {{.*}}@_Znam(i64 noundef 8)
  // EH-NEXT:       br label %while.cond
  // EH:          while.cond:
  // EH-NEXT:       br label %while.body
  // NOEH-NEXT:     br label %while.body
  // CHECK:       while.body:
  // EH-NEXT:       br label %while.cond
  // NOEH-NEXT:     br label %while.body
  new int[2]{({
    while (true)
      ;
    1;
  })};
}

void ClassType() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754279ClassTypeEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 1)
  // EH-NEXT:       br label %while.cond
  // EH:          while.cond:
  // EH-NEXT:       br label %while.body
  // NOEH-NEXT:     br label %while.body
  // CHECK:       while.body:
  // EH-NEXT:       br label %while.cond
  // NOEH-NEXT:     br label %while.body
  // EH:            invoke void @_ZN8GH1754271SC1Ei(ptr {{.*}}[[CALL]], i32 {{.*}})
  // EH-NEXT:       to label %invoke.cont unwind label %lpad
  // EH:          lpad:
  // EH:            call void @_ZdlPvm(ptr noundef [[CALL]], i64 noundef 1)
  new S(({
    while (true)
      ;
    1;
  }));
}

void NothrowNew() {
  // CHECK-LABEL: define dso_local void @_ZN8GH17542710NothrowNewEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm7NoThrow(i64 noundef 4
  // CHECK-NEXT:    [[ISNULL:%.+]] = icmp eq ptr [[CALL]], null
  // CHECK:         br i1 [[ISNULL]], label %new.cont, label %new.notnull
  // CHECK:       new.notnull:
  // EH:            br label %while.cond
  // EH:          while.cond:
  // EH-NEXT:       br label %while.body
  // NOEH-NEXT:     br label %while.body
  // CHECK:       while.body:
  // EH-NEXT:       br label %while.cond
  // NOEH-NEXT:     br label %while.body
  // CHECK:       new.cont:
  // CHECK-NEXT:    phi ptr [ [[CALL]], %{{.*}} ], [ null, %entry ]
  // CHECK-NEXT:    ret void
  new (NoThrow{}) int(({
    while (true)
      ;
    1;
  }));
}

void ReturnFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH17542711ReturnFirstEv()
  // CHECK:         call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    ret void
  new int(({
    return;
    1;
  }));
}

void GotoFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754279GotoFirstEv()
  // CHECK:         call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    br label %L
  // CHECK:       L: ; preds = %entry
  // CHECK-NEXT:    ret void
  new int(({
    goto L;
    1;
  }));
L:;
}

void NoreturnFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH17542713NoreturnFirstEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 4)
  // EH-NEXT:       invoke void @_ZN8GH1754272nrEv()
  // EH-NEXT:       to label %invoke.cont unwind label %lpad
  // EH:          invoke.cont:
  // EH-NEXT:       unreachable
  // EH:          lpad:
  // EH:            call void @_ZdlPvm(ptr noundef [[CALL]], i64 noundef 4)
  // NOEH-NEXT:     call void @_ZN8GH1754272nrEv()
  // NOEH-NEXT:     unreachable
  new int(({
    nr();
    1;
  }));
}

void UnreachableFirst() {
  // CHECK-LABEL: define dso_local void @_ZN8GH17542716UnreachableFirstEv()
  // CHECK:         call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    unreachable
  new int(({
    __builtin_unreachable();
    1;
  }));
}

void Plain() {
  // CHECK-LABEL: define dso_local void @_ZN8GH1754275PlainEv()
  // CHECK:         [[CALL:%.+]] = call {{.*}}@_Znwm(i64 noundef 4)
  // CHECK-NEXT:    store i32 1, ptr [[TMP:%[^,]+]]
  // CHECK-NEXT:    [[V:%.+]] = load i32, ptr [[TMP]]
  // CHECK-NEXT:    store i32 [[V]], ptr [[CALL]]
  // CHECK-NEXT:    ret void
  new int(({ 1; }));
}
}  // namespace GH175427
