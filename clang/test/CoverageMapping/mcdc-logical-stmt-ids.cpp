// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

bool func_if_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  if (a && b)
    if (a && b && c)
      if (a && b && c && d)
        if (a && b && c && d && e)
           if (a && b && c && d && e && f)
              return true;
  return false;
}

// CHECK-LABEL:  Decision,File 0, 4:7 -> 4:13 = M:0, C:2
// CHECK-NEXT:  Branch,File 0, 4:7 -> 4:8 = #2, (#0 - #2) [1,2,0]
// CHECK:  Branch,File 0, 4:12 -> 4:13 = #3, (#2 - #3) [2,0,0]
// CHECK-LABEL:  Decision,File 0, 5:9 -> 5:20 = M:1, C:3
// CHECK-NEXT:  Branch,File 0, 5:9 -> 5:10 = #7, (#1 - #7) [1,3,0]
// CHECK:  Branch,File 0, 5:14 -> 5:15 = #8, (#7 - #8) [3,2,0]
// CHECK:  Branch,File 0, 5:19 -> 5:20 = #6, (#5 - #6) [2,0,0]
// CHECK-LABEL:  Decision,File 0, 6:11 -> 6:27 = M:2, C:4
// CHECK-NEXT:  Branch,File 0, 6:11 -> 6:12 = #14, (#4 - #14) [1,4,0]
// CHECK:  Branch,File 0, 6:16 -> 6:17 = #15, (#14 - #15) [4,3,0]
// CHECK:  Branch,File 0, 6:21 -> 6:22 = #13, (#12 - #13) [3,2,0]
// CHECK:  Branch,File 0, 6:26 -> 6:27 = #11, (#10 - #11) [2,0,0]
// CHECK-LABEL:  Decision,File 0, 7:13 -> 7:34 = M:4, C:5
// CHECK-NEXT:  Branch,File 0, 7:13 -> 7:14 = #23, (#9 - #23) [1,5,0]
// CHECK:  Branch,File 0, 7:18 -> 7:19 = #24, (#23 - #24) [5,4,0]
// CHECK:  Branch,File 0, 7:23 -> 7:24 = #22, (#21 - #22) [4,3,0]
// CHECK:  Branch,File 0, 7:28 -> 7:29 = #20, (#19 - #20) [3,2,0]
// CHECK:  Branch,File 0, 7:33 -> 7:34 = #18, (#17 - #18) [2,0,0]
// CHECK-LABEL:  Decision,File 0, 8:16 -> 8:42 = M:8, C:6
// CHECK-NEXT:  Branch,File 0, 8:16 -> 8:17 = #34, (#16 - #34) [1,6,0]
// CHECK:  Branch,File 0, 8:21 -> 8:22 = #35, (#34 - #35) [6,5,0]
// CHECK:  Branch,File 0, 8:26 -> 8:27 = #33, (#32 - #33) [5,4,0]
// CHECK:  Branch,File 0, 8:31 -> 8:32 = #31, (#30 - #31) [4,3,0]
// CHECK:  Branch,File 0, 8:36 -> 8:37 = #29, (#28 - #29) [3,2,0]
// CHECK:  Branch,File 0, 8:41 -> 8:42 = #27, (#26 - #27) [2,0,0]

bool func_if_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  if (a || b)
    if (a || b || c)
      if (a || b || c || d)
        if (a || b || c || d || e)
           if (a || b || c || d || e || f)
              return true;
  return false;
}

// CHECK-LABEL:  Decision,File 0, 40:7 -> 40:13 = M:0, C:2
// CHECK-NEXT:  Branch,File 0, 40:7 -> 40:8 = (#0 - #2), #2 [1,0,2]
// CHECK:  Branch,File 0, 40:12 -> 40:13 = (#2 - #3), #3 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 41:9 -> 41:20 = M:1, C:3
// CHECK-NEXT:  Branch,File 0, 41:9 -> 41:10 = (#1 - #7), #7 [1,0,3]
// CHECK:  Branch,File 0, 41:14 -> 41:15 = (#7 - #8), #8 [3,0,2]
// CHECK:  Branch,File 0, 41:19 -> 41:20 = (#5 - #6), #6 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 42:11 -> 42:27 = M:2, C:4
// CHECK-NEXT:  Branch,File 0, 42:11 -> 42:12 = (#4 - #14), #14 [1,0,4]
// CHECK:  Branch,File 0, 42:16 -> 42:17 = (#14 - #15), #15 [4,0,3]
// CHECK:  Branch,File 0, 42:21 -> 42:22 = (#12 - #13), #13 [3,0,2]
// CHECK:  Branch,File 0, 42:26 -> 42:27 = (#10 - #11), #11 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 43:13 -> 43:34 = M:4, C:5
// CHECK-NEXT:  Branch,File 0, 43:13 -> 43:14 = (#9 - #23), #23 [1,0,5]
// CHECK:  Branch,File 0, 43:18 -> 43:19 = (#23 - #24), #24 [5,0,4]
// CHECK:  Branch,File 0, 43:23 -> 43:24 = (#21 - #22), #22 [4,0,3]
// CHECK:  Branch,File 0, 43:28 -> 43:29 = (#19 - #20), #20 [3,0,2]
// CHECK:  Branch,File 0, 43:33 -> 43:34 = (#17 - #18), #18 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 44:16 -> 44:42 = M:8, C:6
// CHECK-NEXT:  Branch,File 0, 44:16 -> 44:17 = (#16 - #34), #34 [1,0,6]
// CHECK:  Branch,File 0, 44:21 -> 44:22 = (#34 - #35), #35 [6,0,5]
// CHECK:  Branch,File 0, 44:26 -> 44:27 = (#32 - #33), #33 [5,0,4]
// CHECK:  Branch,File 0, 44:31 -> 44:32 = (#30 - #31), #31 [4,0,3]
// CHECK:  Branch,File 0, 44:36 -> 44:37 = (#28 - #29), #29 [3,0,2]
// CHECK:  Branch,File 0, 44:41 -> 44:42 = (#26 - #27), #27 [2,0,0]

bool func_if_mix(bool a, bool b, bool c, bool d, bool e, bool f) {
  if (a || b)
    if (a && (b || c))
      if ((a || b) && (c || d))
        if (a && (b || c) && (d || e))
          if ((a || b) && (c || d) && (e || f))
            return true;
  return false;
}

// CHECK-LABEL:  Decision,File 0, 76:7 -> 76:13 = M:0, C:2
// CHECK-NEXT:  Branch,File 0, 76:7 -> 76:8 = (#0 - #2), #2 [1,0,2]
// CHECK:  Branch,File 0, 76:12 -> 76:13 = (#2 - #3), #3 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 77:9 -> 77:22 = M:1, C:3
// CHECK-NEXT:  Branch,File 0, 77:9 -> 77:10 = #5, (#1 - #5) [1,2,0]
// CHECK:  Branch,File 0, 77:15 -> 77:16 = (#5 - #6), #6 [2,0,3]
// CHECK:  Branch,File 0, 77:20 -> 77:21 = (#6 - #7), #7 [3,0,0]
// CHECK-LABEL:  Decision,File 0, 78:11 -> 78:31 = M:2, C:4
// CHECK-NEXT:  File 0
// CHECK-NEXT:  Branch,File 0, 78:12 -> 78:13 = (#4 - #10), #10 [1,2,3]
// CHECK:  Branch,File 0, 78:17 -> 78:18 = (#10 - #11), #11 [3,2,0]
// CHECK:  Branch,File 0, 78:24 -> 78:25 = (#9 - #12), #12 [2,0,4]
// CHECK:  Branch,File 0, 78:29 -> 78:30 = (#12 - #13), #13 [4,0,0]
// CHECK-LABEL:  Decision,File 0, 79:13 -> 79:38 = M:4, C:5
// CHECK-NEXT:  Branch,File 0, 79:13 -> 79:14 = #16, (#8 - #16) [1,3,0]
// CHECK:  Branch,File 0, 79:19 -> 79:20 = (#16 - #17), #17 [3,2,4]
// CHECK:  Branch,File 0, 79:24 -> 79:25 = (#17 - #18), #18 [4,2,0]
// CHECK:  Branch,File 0, 79:31 -> 79:32 = (#15 - #19), #19 [2,0,5]
// CHECK:  Branch,File 0, 79:36 -> 79:37 = (#19 - #20), #20 [5,0,0]
// CHECK-LABEL:  Decision,File 0, 80:15 -> 80:47 = M:8, C:6
// CHECK-NEXT:  File 0
// CHECK-NEXT:  Branch,File 0, 80:16 -> 80:17 = (#14 - #24), #24 [1,3,4]
// CHECK:  Branch,File 0, 80:21 -> 80:22 = (#24 - #25), #25 [4,3,0]
// CHECK:  Branch,File 0, 80:28 -> 80:29 = (#23 - #26), #26 [3,2,5]
// CHECK:  Branch,File 0, 80:33 -> 80:34 = (#26 - #27), #27 [5,2,0]
// CHECK:  Branch,File 0, 80:40 -> 80:41 = (#22 - #28), #28 [2,0,6]
// CHECK:  Branch,File 0, 80:45 -> 80:46 = (#28 - #29), #29 [6,0,0]
