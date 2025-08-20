// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int shouldNotGenBranchRet(int x) {
  if (x > 5)
    goto err;
  return 0;
err:
  return -1;
}
// CIR:  cir.func dso_local @_Z21shouldNotGenBranchReti
// CIR:    cir.if {{.*}} {
// CIR:      cir.goto "err"
// CIR:    }
// CIR:    [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:    cir.store [[ZERO]], [[RETVAL:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    [[RET:%.*]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i
// CIR:  ^bb2:
// CIR:    cir.label "err"
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    [[MINUS:%.*]] = cir.unary(minus, [[ONE]]) nsw : !s32i, !s32i
// CIR:    cir.store [[MINUS]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:    cir.br ^bb1

// OGCG: define dso_local noundef i32 @_Z21shouldNotGenBranchReti
// OGCG: if.then:
// OGCG:   br label %err
// OGCG: if.end:
// OGCG:   br label %return
// OGCG: err:
// OGCG:   br label %return
// OGCG: return:

int shouldGenBranch(int x) {
  if (x > 5)
    goto err;
  x++;
err:
  return -1;
}
// CIR:  cir.func dso_local @_Z15shouldGenBranchi
// CIR:    cir.if {{.*}} {
// CIR:      cir.goto "err"
// CIR:    }
// CIR:    cir.br ^bb1
// CIR:  ^bb1:
// CIR:    cir.label "err"

// OGCG: define dso_local noundef i32 @_Z15shouldGenBranchi
// OGCG: if.then:
// OGCG:   br label %err
// OGCG: if.end:
// OGCG:   br label %err
// OGCG: err:
// OGCG:   ret

void severalLabelsInARow(int a) {
  int b = a;
  goto end1;
  b = b + 1;
  goto end2;
end1:
end2:
  b = b + 2;
}
// CIR:  cir.func dso_local @_Z19severalLabelsInARowi
// CIR:    cir.goto "end1"
// CIR:  ^bb[[#BLK1:]]
// CIR:    cir.goto "end2"
// CIR:  ^bb[[#BLK2:]]:
// CIR:    cir.label "end1"
// CIR:    cir.br ^bb[[#BLK3:]]
// CIR:  ^bb[[#BLK3]]:
// CIR:    cir.label "end2"

// OGCG: define dso_local void @_Z19severalLabelsInARowi
// OGCG:   br label %end1
// OGCG: end1:
// OGCG:   br label %end2
// OGCG: end2:
// OGCG:   ret

void severalGotosInARow(int a) {
  int b = a;
  goto end;
  goto end;
end:
  b = b + 2;
}
// CIR:  cir.func dso_local @_Z18severalGotosInARowi
// CIR:    cir.goto "end"
// CIR:  ^bb[[#BLK1:]]:
// CIR:    cir.goto "end"
// CIR:  ^bb[[#BLK2:]]:
// CIR:    cir.label "end"

// OGCG: define dso_local void @_Z18severalGotosInARowi(i32 noundef %a) #0 {
// OGCG:   br label %end
// OGCG: end:
// OGCG:   ret void

extern "C" void action1();
extern "C" void action2();
extern "C" void multiple_non_case(int v) {
  switch (v) {
    default:
        action1();
      l2:
        action2();
        break;
  }
}

// CIR: cir.func dso_local @multiple_non_case
// CIR: cir.switch
// CIR: cir.case(default, []) {
// CIR: cir.call @action1()
// CIR: cir.br ^[[BB1:[a-zA-Z0-9]+]]
// CIR: ^[[BB1]]:
// CIR: cir.label
// CIR: cir.call @action2()
// CIR: cir.break

// OGCG: define dso_local void @multiple_non_case
// OGCG: sw.default:
// OGCG:   call void @action1()
// OGCG:   br label %l2
// OGCG: l2:
// OGCG:   call void @action2()
// OGCG:   br label [[BREAK:%.*]]

extern "C" void case_follow_label(int v) {
  switch (v) {
    case 1:
    label:
    case 2:
      action1();
      break;
    default:
      action2();
      goto label;
  }
}

// CIR: cir.func dso_local @case_follow_label
// CIR: cir.switch
// CIR: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:   cir.label "label"
// CIR: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR:   cir.call @action1()
// CIR:   cir.break
// CIR: cir.case(default, []) {
// CIR:   cir.call @action2()
// CIR:   cir.goto "label"

// OGCG: define dso_local void @case_follow_label
// OGCG: sw.bb:
// OGCG:   br label %label
// OGCG: label:
// OGCG:   br label %sw.bb1
// OGCG: sw.bb1:
// OGCG:   call void @action1()
// OGCG:   br label %sw.epilog
// OGCG: sw.default:
// OGCG:   call void @action2()
// OGCG:   br label %label
// OGCG: sw.epilog:
// OGCG:   ret void

extern "C" void default_follow_label(int v) {
  switch (v) {
    case 1:
    case 2:
      action1();
      break;
    label:
    default:
      action2();
      goto label;
  }
}

// CIR: cir.func dso_local @default_follow_label
// CIR: cir.switch
// CIR: cir.case(equal, [#cir.int<1> : !s32i]) {
// CIR:   cir.yield
// CIR: cir.case(equal, [#cir.int<2> : !s32i]) {
// CIR:   cir.call @action1()
// CIR:   cir.break
// CIR:   cir.label "label"
// CIR: cir.case(default, []) {
// CIR:   cir.call @action2()
// CIR:   cir.goto "label"

// OGCG: define dso_local void @default_follow_label
// OGCG: sw.bb:
// OGCG:   call void @action1()
// OGCG:   br label %sw.epilog
// OGCG: label:
// OGCG:   br label %sw.default
// OGCG: sw.default:
// OGCG:   call void @action2()
// OGCG:   br label %label
// OGCG: sw.epilog:
// OGCG:   ret void
