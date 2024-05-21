// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

enum letter {
 A, B, C, D, E, F, G, H, I, J, L
};

int sw1(enum letter c) {
  switch (c) { 
    case A ... C:
    case D:
    case E ... F:
    case G ... L:
      return 1;
    default: 
      return 0;
  }
}

//      CIR:  cir.func @_Z3sw16letter
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (anyof, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] : !s32i) {
//      CIR:        cir.return
// CIR-NEXT:      },
// CIR-NEXT:      case (default) {
//      CIR:        cir.return
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw16letter
//      LLVM:    switch i32 %[[C:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:      i32 0, label %[[CASE:[0-9]+]]
// LLVM-NEXT:      i32 1, label %[[CASE]]
// LLVM-NEXT:      i32 2, label %[[CASE]]
// LLVM-NEXT:      i32 3, label %[[CASE]]
// LLVM-NEXT:      i32 4, label %[[CASE]]
// LLVM-NEXT:      i32 5, label %[[CASE]]
// LLVM-NEXT:      i32 6, label %[[CASE]]
// LLVM-NEXT:      i32 7, label %[[CASE]]
// LLVM-NEXT:      i32 8, label %[[CASE]]
// LLVM-NEXT:      i32 9, label %[[CASE]]
// LLVM-NEXT:      i32 10, label %[[CASE]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE]]:
//      LLVM:    store i32 1
//      LLVM:    ret
//      LLVM:  [[DEFAULT]]:
//      LLVM:    store i32 0
//      LLVM:    ret


int sw2(enum letter c) {
  switch (c) { 
    case A ... C:
    case L ... A:
      return 1;
    default: 
      return 0;
  }
}

//      CIR:  cir.func @_Z3sw26letter
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (anyof, [0, 1, 2] : !s32i) {
//      CIR:        cir.return
// CIR-NEXT:      },
// CIR-NEXT:      case (default) {
//      CIR:        cir.return
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw26letter
//      LLVM:    switch i32 %[[C:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:      i32 0, label %[[CASE:[0-9]+]]
// LLVM-NEXT:      i32 1, label %[[CASE]]
// LLVM-NEXT:      i32 2, label %[[CASE]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE]]:
//      LLVM:    store i32 1
//      LLVM:    ret
//      LLVM:  [[DEFAULT]]:
//      LLVM:    store i32 0
//      LLVM:    ret

void sw3(enum letter c) {
  int x = 0;
  switch (c) { 
  case A ... C:
    x = 1;
    break;
  case D ... F:
    x = 2;
    break;
  case G ... I:
    x = 3;
    break;
  case J ... L:
    x = 4;
    break;
  }
}

//      CIR:  cir.func @_Z3sw36letter
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (anyof, [0, 1, 2] : !s32i) {
// CIR-NEXT:        cir.int<1>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (anyof, [3, 4, 5] : !s32i) {
// CIR-NEXT:        cir.int<2>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (anyof, [6, 7, 8] : !s32i) {
// CIR-NEXT:        cir.int<3>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (anyof, [9, 10] : !s32i) {
// CIR-NEXT:        cir.int<4>
//      CIR:        cir.break
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw36letter
//      LLVM:    switch i32 %[[C:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:      i32 0, label %[[CASE_AC:[0-9]+]]
// LLVM-NEXT:      i32 1, label %[[CASE_AC]]
// LLVM-NEXT:      i32 2, label %[[CASE_AC]]
// LLVM-NEXT:      i32 3, label %[[CASE_DF:[0-9]+]]
// LLVM-NEXT:      i32 4, label %[[CASE_DF]]
// LLVM-NEXT:      i32 5, label %[[CASE_DF]]
// LLVM-NEXT:      i32 6, label %[[CASE_GI:[0-9]+]]
// LLVM-NEXT:      i32 7, label %[[CASE_GI]]
// LLVM-NEXT:      i32 8, label %[[CASE_GI]]
// LLVM-NEXT:      i32 9, label %[[CASE_JL:[0-9]+]]
// LLVM-NEXT:      i32 10, label %[[CASE_JL]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_AC]]:
//      LLVM:    store i32 1, ptr %[[X:[0-9]+]]
//      LLVM:    br label %[[EPILOG:[0-9]+]]
//      LLVM:  [[CASE_DF]]:
//      LLVM:    store i32 2, ptr %[[X]]
//      LLVM:    br label %[[EPILOG]]
//      LLVM:  [[CASE_GI]]:
//      LLVM:    store i32 3, ptr %[[X]]
//      LLVM:    br label %[[EPILOG]]
//      LLVM:  [[CASE_JL]]:
//      LLVM:    store i32 4, ptr %[[X]]
//      LLVM:    br label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
//      LLVM:    ret void

void sw4(int x) {
  switch (x) {
    case 66 ... 233:
      break;
    case -50 ... 50:
      break;
  }
}

//      CIR:  cir.func @_Z3sw4i
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (anyof, [66, 67, 68, 69, {{[0-9, ]+}}, 230, 231, 232, 233] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (anyof, [-50, -49, -48, -47, {{[0-9, -]+}}, -1, 0, 1, {{[0-9, ]+}}, 47, 48, 49, 50] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw4i
//      LLVM:    switch i32 %[[X:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:      i32 66, label %[[CASE_66_233:[0-9]+]]
// LLVM-NEXT:      i32 67, label %[[CASE_66_233]]
//                 ...
//      LLVM:      i32 232, label %[[CASE_66_233]]
// LLVM-NEXT:      i32 233, label %[[CASE_66_233]]
// LLVM-NEXT:      i32 -50, label %[[CASE_NEG50_50:[0-9]+]]
// LLVM-NEXT:      i32 -49, label %[[CASE_NEG50_50]]
//                 ...
//      LLVM:      i32 -1, label %[[CASE_NEG50_50]]
// LLVM-NEXT:      i32 0, label %[[CASE_NEG50_50]]
// LLVM-NEXT:      i32 1, label %[[CASE_NEG50_50]]
//                 ...
//      LLVM:      i32 49, label %[[CASE_NEG50_50]]
// LLVM-NEXT:      i32 50, label %[[CASE_NEG50_50]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_66_233]]:
//      LLVM:    br label %[[EPILOG:[0-9]+]]
//      LLVM:  [[CASE_NEG50_50]]:
//      LLVM:    br label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
//      LLVM:    ret void


