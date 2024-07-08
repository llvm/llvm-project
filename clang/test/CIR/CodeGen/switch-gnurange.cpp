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
// CIR-NEXT:      case (range, [0, 2] : !s32i) {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [4, 5] : !s32i) {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [6, 10] : !s32i) {
// CIR-NEXT:        cir.yield
// CIR-NEXT:      },
// CIR-NEXT:      case (equal, 3) {
// CIR-NEXT:        cir.int<1>
//      CIR:        cir.return
// CIR-NEXT:      },
// CIR-NEXT:      case (default) {
// CIR-NEXT:        cir.int<0>
//      CIR:        cir.return
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw16letter
//      LLVM:    switch i32 %[[C:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:      i32 3, label %[[CASE_3:[0-9]+]]
// LLVM-NEXT:      i32 0, label %[[CASE_0_2:[0-9]+]]
// LLVM-NEXT:      i32 1, label %[[CASE_0_2]]
// LLVM-NEXT:      i32 2, label %[[CASE_0_2]]
// LLVM-NEXT:      i32 4, label %[[CASE_4_5:[0-9]+]]
// LLVM-NEXT:      i32 5, label %[[CASE_4_5]]
// LLVM-NEXT:      i32 6, label %[[CASE_6_10:[0-9]+]]
// LLVM-NEXT:      i32 7, label %[[CASE_6_10]]
// LLVM-NEXT:      i32 8, label %[[CASE_6_10]]
// LLVM-NEXT:      i32 9, label %[[CASE_6_10]]
// LLVM-NEXT:      i32 10, label %[[CASE_6_10]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_0_2]]:
//      LLVM:    br label %[[CASE_4_5]]
//      LLVM:  [[CASE_4_5]]:
//      LLVM:    br label %[[CASE_6_10]]
//      LLVM:  [[CASE_6_10]]:
//      LLVM:    br label %[[CASE_3]]
//      LLVM:  [[CASE_3]]:
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
// CIR-NEXT:      case (range, [0, 2] : !s32i) {
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
// CIR-NEXT:      case (range, [0, 2] : !s32i) {
// CIR-NEXT:        cir.int<1>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [3, 5] : !s32i) {
// CIR-NEXT:        cir.int<2>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [6, 8] : !s32i) {
// CIR-NEXT:        cir.int<3>
//      CIR:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [9, 10] : !s32i) {
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
// LLVM-NEXT:    br label %[[EPILOG_END:[0-9]+]]
//      LLVM:  [[EPILOG_END]]:
// LLVM-NEXT:    ret void

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
// CIR-NEXT:      case (range, [66, 233] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [-50, 50] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      }
// CIR-NEXT:      ]
// CIR-NEXT:    }

//      LLVM:  @_Z3sw4i
//      LLVM:    switch i32 %[[X:[0-9]+]], label %[[JUDGE_NEG50_50:[0-9]+]] [
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_66_233:[0-9]+]]:
// LLVM-NEXT:    br label %[[EPILOG:[0-9]+]]
//      LLVM:  [[CASE_NEG50_50:[0-9]+]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[JUDGE_NEG50_50]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], -50
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], 100
// LLVM-NEXT:    br i1 %[[DIFF_CMP]], label %[[CASE_NEG50_50]], label %[[JUDGE_66_233:[0-9]+]]
//      LLVM:  [[JUDGE_66_233]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], 66
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], 167
//      LLVM:    br i1 %[[DIFF_CMP]], label %[[CASE_66_233]], label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
// LLVM-NEXT:    br label %[[EPILOG_END:[0-9]+]]
//      LLVM:  [[EPILOG_END]]:
// LLVM-NEXT:    ret void

void sw5(int x) {
  int y = 0;
  switch (x) {
    case 100 ... -100:
      y = 1;
  }
}

//      CIR:  cir.func @_Z3sw5i
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (range, [100, -100] : !s32i) {
// CIR-NEXT:        cir.int<1>
//      CIR:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      ]

//      LLVM:  @_Z3sw5i
//      LLVM:    switch i32 %[[X:[0-9]+]], label %[[EPILOG:[0-9]+]] [
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_100_NEG100:[0-9]+]]:
// LLVM-NEXT:    store i32 1, ptr %[[Y:[0-9]+]]
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
// LLVM-NEXT:    br label %[[EPILOG_END:[0-9]+]]
//      LLVM:  [[EPILOG_END]]:
// LLVM-NEXT:    ret void

void sw6(int x) {
  int y = 0;
  switch (x) {
    case -2147483648 ... 2147483647:
      y = 1;
  }
}

//      CIR:  cir.func @_Z3sw6i
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (range, [-2147483648, 2147483647] : !s32i) {
// CIR-NEXT:        cir.int<1>
//      CIR:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:      ]

//      LLVM:  @_Z3sw6i
//      LLVM:    switch i32 %[[X:[0-9]+]], label %[[DEFAULT:[0-9]+]] [
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_MIN_MAX:[0-9]+]]:
// LLVM-NEXT:    store i32 1, ptr %[[Y:[0-9]+]]
// LLVM-NEXT:    br label %[[EPILOG:[0-9]+]]
//      LLVM:  [[DEFAULT]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], -2147483648
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], -1
// LLVM-NEXT:    br i1 %[[DIFF_CMP]], label %[[CASE_MIN_MAX]], label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
// LLVM-NEXT:    br label %[[EPILOG_END:[0-9]+]]
//      LLVM:  [[EPILOG_END]]:
// LLVM-NEXT:    ret void

void sw7(int x) {
  switch(x) {
  case 0:
    break;
  case 100 ... 200:
    break;
  case 1:
    break;
  case 300 ... 400:
    break;
  default:
    break;
  case 500 ... 600:
    break;
  }
}

//      CIR:  cir.func @_Z3sw7i
//      CIR:    cir.scope {
//      CIR:      cir.switch
// CIR-NEXT:      case (equal, 0) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [100, 200] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (equal, 1) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [300, 400] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (default) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      },
// CIR-NEXT:      case (range, [500, 600] : !s32i) {
// CIR-NEXT:        cir.break
// CIR-NEXT:      }

//      LLVM:  @_Z3sw7i
//      LLVM:    switch i32 %[[X:[0-9]+]], label %[[JUDGE_RANGE_500_600:[0-9]+]] [
// LLVM-NEXT:      i32 0, label %[[CASE_0:[0-9]+]]
// LLVM-NEXT:      i32 1, label %[[CASE_1:[0-9]+]]
// LLVM-NEXT:    ]
//      LLVM:  [[CASE_0]]:
// LLVM-NEXT:    br label %[[EPILOG:[0-9]+]]
//      LLVM:  [[CASE_100_200:[0-9]+]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[CASE_1]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[CASE_300_400:[0-9]+]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[JUDGE_RANGE_500_600]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], 500
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], 100
// LLVM-NEXT:    br i1 %[[DIFF_CMP]], label %[[CASE_500_600:[0-9]+]], label %[[JUDGE_RANGE_300_400:[0-9]+]]
//      LLVM:  [[JUDGE_RANGE_300_400]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], 300
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], 100
// LLVM-NEXT:    br i1 %[[DIFF_CMP]], label %[[CASE_300_400]], label %[[JUDGE_RANGE_100_200:[0-9]+]]
//      LLVM:  [[JUDGE_RANGE_100_200]]:
// LLVM-NEXT:    %[[DIFF:[0-9]+]] = sub i32 %[[X]], 100
// LLVM-NEXT:    %[[DIFF_CMP:[0-9]+]] = icmp ule i32 %[[DIFF]], 100
// LLVM-NEXT:    br i1 %[[DIFF_CMP]], label %[[CASE_100_200]], label %[[DEFAULT:[0-9]+]]
//      LLVM:  [[DEFAULT]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[CASE_500_600]]:
// LLVM-NEXT:    br label %[[EPILOG]]
//      LLVM:  [[EPILOG]]:
// LLVM-NEXT:    br label %[[EPILOG_END:[0-9]+]]
//      LLVM:  [[EPILOG_END]]:
// LLVM-NEXT:    ret void

