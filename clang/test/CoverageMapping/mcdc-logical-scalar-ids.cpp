// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

extern bool bar (bool, bool, bool, bool, bool);
bool func_scalar_and(bool a, bool b, bool c, bool d, bool e, bool f) {
    bool res1 = a && b;
    bool res2 = a && b && c;
    bool res3 = a && b && c && d;
    bool res4 = a && b && c && d && e;
    bool res5 = a && b && c && d && e && f;
    return bar(res1, res2, res3, res4, res5);
}

// CHECK-LABEL: Decision,File 0, 5:17 -> 5:23 = M:0, C:2
// CHECK-NEXT: Branch,File 0, 5:17 -> 5:18 = #1, (#0 - #1) [1,2,0]
// CHECK: Branch,File 0, 5:22 -> 5:23 = #2, (#1 - #2) [2,0,0]
// CHECK-LABEL: Decision,File 0, 6:17 -> 6:28 = M:1, C:3
// CHECK-NEXT: Branch,File 0, 6:17 -> 6:18 = #5, (#0 - #5) [1,3,0]
// CHECK: Branch,File 0, 6:22 -> 6:23 = #6, (#5 - #6) [3,2,0]
// CHECK: Branch,File 0, 6:27 -> 6:28 = #4, (#3 - #4) [2,0,0]
// CHECK-LABEL: Decision,File 0, 7:17 -> 7:33 = M:2, C:4
// CHECK-NEXT: Branch,File 0, 7:17 -> 7:18 = #11, (#0 - #11) [1,4,0]
// CHECK: Branch,File 0, 7:22 -> 7:23 = #12, (#11 - #12) [4,3,0]
// CHECK: Branch,File 0, 7:27 -> 7:28 = #10, (#9 - #10) [3,2,0]
// CHECK: Branch,File 0, 7:32 -> 7:33 = #8, (#7 - #8) [2,0,0]
// CHECK-LABEL: Decision,File 0, 8:17 -> 8:38 = M:4, C:5
// CHECK-NEXT: Branch,File 0, 8:17 -> 8:18 = #19, (#0 - #19) [1,5,0]
// CHECK: Branch,File 0, 8:22 -> 8:23 = #20, (#19 - #20) [5,4,0]
// CHECK: Branch,File 0, 8:27 -> 8:28 = #18, (#17 - #18) [4,3,0]
// CHECK: Branch,File 0, 8:32 -> 8:33 = #16, (#15 - #16) [3,2,0]
// CHECK: Branch,File 0, 8:37 -> 8:38 = #14, (#13 - #14) [2,0,0]
// CHECK-LABEL: Decision,File 0, 9:17 -> 9:43 = M:8, C:6
// CHECK-NEXT: Branch,File 0, 9:17 -> 9:18 = #29, (#0 - #29) [1,6,0]
// CHECK: Branch,File 0, 9:22 -> 9:23 = #30, (#29 - #30) [6,5,0]
// CHECK: Branch,File 0, 9:27 -> 9:28 = #28, (#27 - #28) [5,4,0]
// CHECK: Branch,File 0, 9:32 -> 9:33 = #26, (#25 - #26) [4,3,0]
// CHECK: Branch,File 0, 9:37 -> 9:38 = #24, (#23 - #24) [3,2,0]
// CHECK: Branch,File 0, 9:42 -> 9:43 = #22, (#21 - #22) [2,0,0]

bool func_scalar_or(bool a, bool b, bool c, bool d, bool e, bool f) {
    bool res1 = a || b;
    bool res2 = a || b || c;
    bool res3 = a || b || c || d;
    bool res4 = a || b || c || d || e;
    bool res5 = a || b || c || d || e || f;
    return bar(res1, res2, res3, res4, res5);
}

// CHECK-LABEL: Decision,File 0, 40:17 -> 40:23 = M:0, C:2
// CHECK-NEXT: Branch,File 0, 40:17 -> 40:18 = (#0 - #1), #1 [1,0,2]
// CHECK: Branch,File 0, 40:22 -> 40:23 = (#1 - #2), #2 [2,0,0]
// CHECK-LABEL: Decision,File 0, 41:17 -> 41:28 = M:1, C:3
// CHECK-NEXT: Branch,File 0, 41:17 -> 41:18 = (#0 - #5), #5 [1,0,3]
// CHECK: Branch,File 0, 41:22 -> 41:23 = (#5 - #6), #6 [3,0,2]
// CHECK: Branch,File 0, 41:27 -> 41:28 = (#3 - #4), #4 [2,0,0]
// CHECK-LABEL: Decision,File 0, 42:17 -> 42:33 = M:2, C:4
// CHECK-NEXT: Branch,File 0, 42:17 -> 42:18 = (#0 - #11), #11 [1,0,4]
// CHECK: Branch,File 0, 42:22 -> 42:23 = (#11 - #12), #12 [4,0,3]
// CHECK: Branch,File 0, 42:27 -> 42:28 = (#9 - #10), #10 [3,0,2]
// CHECK: Branch,File 0, 42:32 -> 42:33 = (#7 - #8), #8 [2,0,0]
// CHECK-LABEL: Decision,File 0, 43:17 -> 43:38 = M:4, C:5
// CHECK-NEXT: Branch,File 0, 43:17 -> 43:18 = (#0 - #19), #19 [1,0,5]
// CHECK: Branch,File 0, 43:22 -> 43:23 = (#19 - #20), #20 [5,0,4]
// CHECK: Branch,File 0, 43:27 -> 43:28 = (#17 - #18), #18 [4,0,3]
// CHECK: Branch,File 0, 43:32 -> 43:33 = (#15 - #16), #16 [3,0,2]
// CHECK: Branch,File 0, 43:37 -> 43:38 = (#13 - #14), #14 [2,0,0]
// CHECK-LABEL: Decision,File 0, 44:17 -> 44:43 = M:8, C:6
// CHECK-NEXT: Branch,File 0, 44:17 -> 44:18 = (#0 - #29), #29 [1,0,6]
// CHECK: Branch,File 0, 44:22 -> 44:23 = (#29 - #30), #30 [6,0,5]
// CHECK: Branch,File 0, 44:27 -> 44:28 = (#27 - #28), #28 [5,0,4]
// CHECK: Branch,File 0, 44:32 -> 44:33 = (#25 - #26), #26 [4,0,3]
// CHECK: Branch,File 0, 44:37 -> 44:38 = (#23 - #24), #24 [3,0,2]
// CHECK: Branch,File 0, 44:42 -> 44:43 = (#21 - #22), #22 [2,0,0]


bool func_scalar_mix(bool a, bool b, bool c, bool d, bool e, bool f) {
    bool res1 = a || b;
    bool res2 = a && (b || c);
    bool res3 = (a || b) && (c || d);
    bool res4 = a && (b || c) && (d || e);
    bool res5 = (a || b) && (c || d) && (e || f);
    return bar(res1, res2, res3, res4, res5);
}

// CHECK-LABEL:  Decision,File 0, 76:17 -> 76:23 = M:0, C:2
// CHECK-NEXT:  Branch,File 0, 76:17 -> 76:18 = (#0 - #1), #1 [1,0,2]
// CHECK:  Branch,File 0, 76:22 -> 76:23 = (#1 - #2), #2 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 77:17 -> 77:30 = M:1, C:3
// CHECK-NEXT:  Branch,File 0, 77:17 -> 77:18 = #3, (#0 - #3) [1,2,0]
// CHECK:  Branch,File 0, 77:23 -> 77:24 = (#3 - #4), #4 [2,0,3]
// CHECK:  Branch,File 0, 77:28 -> 77:29 = (#4 - #5), #5 [3,0,0]
// CHECK-LABEL:  Decision,File 0, 78:17 -> 78:37 = M:2, C:4
// CHECK-NEXT:  File 0
// CHECK-NEXT:  Branch,File 0, 78:18 -> 78:19 = (#0 - #7), #7 [1,2,3]
// CHECK:  Branch,File 0, 78:23 -> 78:24 = (#7 - #8), #8 [3,2,0]
// CHECK:  Branch,File 0, 78:30 -> 78:31 = (#6 - #9), #9 [2,0,4]
// CHECK:  Branch,File 0, 78:35 -> 78:36 = (#9 - #10), #10 [4,0,0]
// CHECK-LABEL:  Decision,File 0, 79:17 -> 79:42 = M:4, C:5
// CHECK-NEXT:  Branch,File 0, 79:17 -> 79:18 = #12, (#0 - #12) [1,3,0]
// CHECK:  Branch,File 0, 79:23 -> 79:24 = (#12 - #13), #13 [3,2,4]
// CHECK:  Branch,File 0, 79:28 -> 79:29 = (#13 - #14), #14 [4,2,0]
// CHECK:  Branch,File 0, 79:35 -> 79:36 = (#11 - #15), #15 [2,0,5]
// CHECK:  Branch,File 0, 79:40 -> 79:41 = (#15 - #16), #16 [5,0,0]
// CHECK-LABEL:  Decision,File 0, 80:17 -> 80:49 = M:8, C:6
// CHECK-NEXT:  File 0
// CHECK-NEXT:  Branch,File 0, 80:18 -> 80:19 = (#0 - #19), #19 [1,3,4]
// CHECK:  Branch,File 0, 80:23 -> 80:24 = (#19 - #20), #20 [4,3,0]
// CHECK:  Branch,File 0, 80:30 -> 80:31 = (#18 - #21), #21 [3,2,5]
// CHECK:  Branch,File 0, 80:35 -> 80:36 = (#21 - #22), #22 [5,2,0]
// CHECK:  Branch,File 0, 80:42 -> 80:43 = (#17 - #23), #23 [2,0,6]
// CHECK:  Branch,File 0, 80:47 -> 80:48 = (#23 - #24), #24 [6,0,0]
