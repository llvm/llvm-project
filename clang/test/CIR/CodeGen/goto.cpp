// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat %s -o %t1.cir
// RUN: FileCheck --input-file=%t1.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t2.cir
// RUN: FileCheck --input-file=%t2.cir %s -check-prefix=NOFLAT


void g0(int a) {
  int b = a;
  goto end;
  b = b + 1;
end:
  b = b + 2;
}

// CHECK:   cir.func @_Z2g0i
// CHECK-NEXT  %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// CHECK-NEXT  cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT  %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT  cir.store %2, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT  cir.br ^bb2
// CHECK-NEXT ^bb1:  // no predecessors
// CHECK-NEXT   %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT   %4 = cir.const 1 : !s32i
// CHECK-NEXT   %5 = cir.binop(add, %3, %4) : !s32i
// CHECK-NEXT   cir.store %5, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT   cir.br ^bb2
// CHECK-NEXT ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT   %6 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT   %7 = cir.const 2 : !s32i
// CHECK-NEXT   %8 = cir.binop(add, %6, %7) : !s32i
// CHECK-NEXT   cir.store %8, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT   cir.return

void g1(int a) {
  int x = 0;
  goto end;
end:
  int y = a + 2;
}

// Make sure alloca for "y" shows up in the entry block
// CHECK: cir.func @_Z2g1i(%arg0: !s32i
// CHECK-NEXT: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>

int g2() {
  int b = 1;
  goto end;
  b = b + 1;
end:
  b = b + 2;
  return 1;
}

// Make sure (1) we don't get dangling unused cleanup blocks
//           (2) generated returns consider the function type

// CHECK: cir.func @_Z2g2v() -> !s32i

// CHECK:     cir.br ^bb2
// CHECK-NEXT:   ^bb1:  // no predecessors
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1

// CHECK:     [[R:%[0-9]+]] = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:     [[R]] : !s32i
// CHECK-NEXT:   }


int shouldNotGenBranchRet(int x) {
  if (x > 5)
    goto err;
  return 0;
err:
  return -1;
}
// NOFLAT:  cir.func @_Z21shouldNotGenBranchReti
// NOFLAT:    cir.if %8 {
// NOFLAT:      cir.goto "err"
// NOFLAT:    }
// NOFLAT:  ^bb1:
// NOFLAT:    %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// NOFLAT:    cir.return %3 : !s32i
// NOFLAT:  ^bb2:  // no predecessors
// NOFLAT:    cir.label "err"

int shouldGenBranch(int x) {
  if (x > 5)
    goto err;
  x++;
err:
  return -1;
}
// NOFLAT:  cir.func @_Z15shouldGenBranchi
// NOFLAT:    cir.if %9 {
// NOFLAT:      cir.goto "err"
// NOFLAT:    }
// NOFLAT:    cir.br ^bb1
// NOFLAT:  ^bb1:  
// NOFLAT:    cir.label "err"

int shouldCreateBlkForGoto(int a) {
  switch (a) {
    case(42):
      break;
      goto exit;
    default:
      return 0;
  };

exit:
  return -1;

}
// NOFLAT: cir.func @_Z22shouldCreateBlkForGotoi
// NOFLAT:   case (equal, 42) {
// NOFLAT:     cir.break
// NOFLAT:   ^bb1:  // no predecessors
// NOFLAT:     cir.goto "exit"
// NOFLAT:   }

void severalLabelsInARow(int a) {
  int b = a;
  goto end1;
  b = b + 1;
  goto end2;
end1:
end2:
  b = b + 2;
}
// NOFLAT:  cir.func @_Z19severalLabelsInARowi
// NOFLAT:  ^bb[[#BLK1:]]:
// NOFLAT:    cir.label "end1"
// NOFLAT:    cir.br ^bb[[#BLK2:]]
// NOFLAT:  ^bb[[#BLK2]]:
// NOFLAT:    cir.label "end2"

void severalGotosInARow(int a) {
  int b = a;
  goto end;
  goto end;
end:
  b = b + 2;
}
// NOFLAT:  cir.func @_Z18severalGotosInARowi
// NOFLAT:    cir.goto "end"
// NOFLAT:  ^bb[[#BLK1:]]:
// NOFLAT:    cir.goto "end"
// NOFLAT:  ^bb[[#BLK2:]]:
// NOFLAT:    cir.label "end"


void labelWithoutMatch() {
end:
  return;
}
// NOFLAT:  cir.func @_Z17labelWithoutMatchv()
// NOFLAT:    cir.label "end"
// NOFLAT:    cir.return
// NOFLAT:  }


int jumpIntoLoop(int* ar) {

  if (ar)
    goto label;
  return -1;
  
  while (ar) {
  label:
    ++ar;
  }

  return 0;
}

// CHECK:  cir.func @_Z12jumpIntoLoopPi
// CHECK:    cir.brcond {{.*}} ^bb[[#BLK2:]], ^bb[[#BLK3:]]
// CHECK:  ^bb[[#BLK2]]:
// CHECK:    cir.br ^bb[[#BODY:]]
// CHECK:  ^bb[[#BLK3]]:
// CHECK:    cir.br ^bb[[#BLK4:]]
// CHECK:  ^bb[[#BLK4]]:
// CHECK:    cir.br ^bb[[#RETURN:]]
// CHECK:  ^bb[[#RETURN]]:
// CHECK:    cir.return
// CHECK:  ^bb[[#BLK5:]]:
// CHECK:    cir.br ^bb[[#BLK6:]]
// CHECK:  ^bb[[#BLK6]]:
// CHECK:    cir.br ^bb[[#COND:]]
// CHECK:  ^bb[[#COND]]:
// CHECK:    cir.brcond {{.*}} ^bb[[#BODY]], ^bb[[#EXIT:]]
// CHECK:  ^bb[[#BODY]]: 
// CHECK:    cir.br ^bb[[#COND]]
// CHECK:  ^bb[[#EXIT]]:
// CHECK:    cir.br ^bb[[#BLK7:]]
// CHECK:  ^bb[[#BLK7]]:
// CHECK:    cir.br ^bb[[#RETURN]]



int jumpFromLoop(int* ar) {

  if (!ar) {
err:
    return -1;
}

  while (ar) {
    if (*ar == 42)
      goto err;
    ++ar;
  }
  
  return 0;
}
// CHECK:  cir.func @_Z12jumpFromLoopPi
// CHECK:    cir.brcond {{.*}} ^bb[[#RETURN1:]], ^bb[[#BLK3:]]
// CHECK:  ^bb[[#RETURN1]]:
// CHECK:    cir.return
// CHECK:  ^bb[[#BLK3]]:
// CHECK:    cir.br ^bb[[#BLK4:]]
// CHECK:  ^bb[[#BLK4]]:
// CHECK:    cir.br ^bb[[#BLK5:]]
// CHECK:  ^bb[[#BLK5]]:
// CHECK:    cir.br ^bb[[#COND:]]
// CHECK:  ^bb[[#COND]]: 
// CHECK:    cir.brcond {{.*}} ^bb[[#BODY:]], ^bb[[#EXIT:]]
// CHECK:  ^bb[[#BODY]]:
// CHECK:    cir.br ^bb[[#IF42:]]
// CHECK:  ^bb[[#IF42]]:
// CHECK:    cir.brcond {{.*}} ^bb[[#IF42TRUE:]], ^bb[[#IF42FALSE:]]
// CHECK:  ^bb[[#IF42TRUE]]:
// CHECK:    cir.br ^bb[[#RETURN1]]
// CHECK:  ^bb[[#IF42FALSE]]:
// CHECK:    cir.br ^bb[[#BLK11:]]
// CHECK:  ^bb[[#BLK11]]:
// CHECK:    cir.br ^bb[[#COND]]
// CHECK:  ^bb[[#EXIT]]:
// CHECK:    cir.br ^bb[[#RETURN2:]]
// CHECK:  ^bb[[#RETURN2]]:
// CHECK:    cir.return 
  

void flatLoopWithNoTerminatorInFront(int* ptr) {
  
  if (ptr)
    goto loop;

  do {
    if (!ptr)
      goto end;
  loop:
      ptr++;
  } while(ptr);

  end:
  ;
}

// CHECK:  cir.func @_Z31flatLoopWithNoTerminatorInFrontPi
// CHECK:    cir.brcond {{.*}} ^bb[[#BLK2:]], ^bb[[#BLK3:]]
// CHECK:  ^bb[[#BLK2]]:
// CHECK:    cir.br ^bb[[#LABEL_LOOP:]]
// CHECK:  ^bb[[#BLK3]]:
// CHECK:    cir.br ^bb[[#BLK4:]] 
// CHECK:  ^bb[[#BLK4]]:
// CHECK:    cir.br ^bb[[#BLK5:]]
// CHECK:  ^bb[[#BLK5]]:
// CHECK:    cir.br ^bb[[#BODY:]]
// CHECK:  ^bb[[#COND]]: 
// CHECK:    cir.brcond {{.*}} ^bb[[#BODY]], ^bb[[#EXIT:]]
// CHECK:  ^bb[[#BODY]]:
// CHECK:    cir.br ^bb[[#BLK8:]]
// CHECK:  ^bb[[#BLK8]]:
// CHECK:    cir.brcond {{.*}} ^bb[[#BLK9:]], ^bb[[#BLK10:]]
// CHECK:  ^bb[[#BLK9]]:
// CHECK:    cir.br ^bb[[#RETURN:]]
// CHECK:  ^bb[[#BLK10]]:
// CHECK:    cir.br ^bb[[#BLK11:]]
// CHECK:  ^bb[[#BLK11]]:
// CHECK:    cir.br ^bb[[#LABEL_LOOP]]
// CHECK:  ^bb[[#LABEL_LOOP]]:
// CHECK:    cir.br ^bb[[#COND]]
// CHECK:  ^bb[[#EXIT]]:
// CHECK:    cir.br ^bb[[#BLK14:]]
// CHECK:  ^bb[[#BLK14]]:
// CHECK:    cir.br ^bb[[#RETURN]]
// CHECK:  ^bb[[#RETURN]]:
// CHECK:    cir.return
// CHECK:  }
// CHECK:}