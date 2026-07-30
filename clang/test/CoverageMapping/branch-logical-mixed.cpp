// Test to ensure that each branch condition has an associated branch region
// with expected True/False counters.

// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-logical-mixed.cpp %s | FileCheck %s




bool func() {                        // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+55]]:2 = #0
  bool bt0 = true;
  bool bt1 = true;
  bool bt2 = true;
  bool bt3 = true;
  bool bt4 = true;
  bool bt5 = true;
  bool bf0 = false;
  bool bf1 = false;
  bool bf2 = false;
  bool bf3 = false;
  bool bf4 = false;
  bool bf5 = false;

  bool a = bt0 &&                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #9, (#0 - #9)
           bf0 &&                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #10, (#9 - #10)
           bt1 &&                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #8, (#7 - #8)
           bf1 &&                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #6, (#5 - #6)
           bt2 &&                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #4, (#3 - #4)
           bf2;                     // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = #2, (#1 - #2)

  bool b = bt0 ||                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#0 - #19), #19
           bf0 ||                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#19 - #20), #20
           bt1 ||                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#17 - #18), #18
           bf1 ||                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#15 - #16), #16
           bt2 ||                   // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#13 - #14), #14
           bf2;                     // CHECK: Branch,File 0, [[@LINE]]:12 -> [[@LINE]]:15 = (#11 - #12), #12

  bool c = (bt0  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #26, (#0 - #26)
            bf0) ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #27, (#26 - #27)
           (bt1  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #28, (#25 - #28)
            bf1) ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #29, (#28 - #29)
           (bt2  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #30, (#24 - #30)
            bf2) ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #31, (#30 - #31)
           (bt3  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #32, (#23 - #32)
            bf3) ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #33, (#32 - #33)
           (bt4  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #34, (#22 - #34)
            bf4) ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #35, (#34 - #35)
           (bf5  &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #36, (#21 - #36)
            bf5);                   // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = #37, (#36 - #37)

  bool d = (bt0  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#0 - #43), #43
            bf0) &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#43 - #44), #44
           (bt1  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#42 - #45), #45
            bf1) &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#45 - #46), #46
           (bt2  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#41 - #47), #47
            bf2) &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#47 - #48), #48
           (bt3  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#40 - #49), #49
            bf3) &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#49 - #50), #50
           (bt4  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#39 - #51), #51
            bf4) &&                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#51 - #52), #52
           (bt5  ||                 // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#38 - #53), #53
            bf5);                   // CHECK: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:16 = (#53 - #54), #54

  return a && b && c && d;
}
