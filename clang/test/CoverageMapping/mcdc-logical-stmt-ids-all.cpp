// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

bool func_if_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  if (a && b && c && d && e && f)
    return true;
  return false;
}

// CHECK-LABEL:  Decision,File 0, 4:7 -> 4:33 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 4:7 -> 4:8 = #10, (#0 - #10) [1,6,0]
// CHECK:  Branch,File 0, 4:12 -> 4:13 = #11, (#10 - #11) [6,5,0]
// CHECK:  Branch,File 0, 4:17 -> 4:18 = #9, (#8 - #9) [5,4,0]
// CHECK:  Branch,File 0, 4:22 -> 4:23 = #7, (#6 - #7) [4,3,0]
// CHECK:  Branch,File 0, 4:27 -> 4:28 = #5, (#4 - #5) [3,2,0]
// CHECK:  Branch,File 0, 4:32 -> 4:33 = #3, (#2 - #3) [2,0,0]

bool func_if_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  if (a || b || c || d || e || f)
    return true;
  return false;
}

// CHECK-LABEL:  Decision,File 0, 18:7 -> 18:33 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 18:7 -> 18:8 = (#0 - #10), #10 [1,0,6]
// CHECK:  Branch,File 0, 18:12 -> 18:13 = (#10 - #11), #11 [6,0,5]
// CHECK:  Branch,File 0, 18:17 -> 18:18 = (#8 - #9), #9 [5,0,4]
// CHECK:  Branch,File 0, 18:22 -> 18:23 = (#6 - #7), #7 [4,0,3]
// CHECK:  Branch,File 0, 18:27 -> 18:28 = (#4 - #5), #5 [3,0,2]
// CHECK:  Branch,File 0, 18:32 -> 18:33 = (#2 - #3), #3 [2,0,0]

bool func_while_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  while (a && b && c && d && e && f) { return true; }
  return false;
}

// CHECK-LABEL:  Decision,File 0, 32:10 -> 32:36 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 32:10 -> 32:11 = #10, (#0 - #10) [1,6,0]
// CHECK:  Branch,File 0, 32:15 -> 32:16 = #11, (#10 - #11) [6,5,0]
// CHECK:  Branch,File 0, 32:20 -> 32:21 = #9, (#8 - #9) [5,4,0]
// CHECK:  Branch,File 0, 32:25 -> 32:26 = #7, (#6 - #7) [4,3,0]
// CHECK:  Branch,File 0, 32:30 -> 32:31 = #5, (#4 - #5) [3,2,0]
// CHECK:  Branch,File 0, 32:35 -> 32:36 = #3, (#2 - #3) [2,0,0]

bool func_while_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  while (a || b || c || d || e || f) { return true; }
  return false;
}

// CHECK-LABEL:  Decision,File 0, 45:10 -> 45:36 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 45:10 -> 45:11 = (#0 - #10), #10 [1,0,6]
// CHECK:  Branch,File 0, 45:15 -> 45:16 = (#10 - #11), #11 [6,0,5]
// CHECK:  Branch,File 0, 45:20 -> 45:21 = (#8 - #9), #9 [5,0,4]
// CHECK:  Branch,File 0, 45:25 -> 45:26 = (#6 - #7), #7 [4,0,3]
// CHECK:  Branch,File 0, 45:30 -> 45:31 = (#4 - #5), #5 [3,0,2]
// CHECK:  Branch,File 0, 45:35 -> 45:36 = (#2 - #3), #3 [2,0,0]

bool func_for_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  for (;a && b && c && d && e && f;) { return true; }
  return false;
}

// CHECK-LABEL:  Decision,File 0, 58:9 -> 58:35 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 58:9 -> 58:10 = #10, (#0 - #10) [1,6,0]
// CHECK:  Branch,File 0, 58:14 -> 58:15 = #11, (#10 - #11) [6,5,0]
// CHECK:  Branch,File 0, 58:19 -> 58:20 = #9, (#8 - #9) [5,4,0]
// CHECK:  Branch,File 0, 58:24 -> 58:25 = #7, (#6 - #7) [4,3,0]
// CHECK:  Branch,File 0, 58:29 -> 58:30 = #5, (#4 - #5) [3,2,0]
// CHECK:  Branch,File 0, 58:34 -> 58:35 = #3, (#2 - #3) [2,0,0]

bool func_for_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  for (;a || b || c || d || e || f;) { return true; }
  return false;
}

// CHECK-LABEL:  Decision,File 0, 71:9 -> 71:35 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 71:9 -> 71:10 = (#0 - #10), #10 [1,0,6]
// CHECK:  Branch,File 0, 71:14 -> 71:15 = (#10 - #11), #11 [6,0,5]
// CHECK:  Branch,File 0, 71:19 -> 71:20 = (#8 - #9), #9 [5,0,4]
// CHECK:  Branch,File 0, 71:24 -> 71:25 = (#6 - #7), #7 [4,0,3]
// CHECK:  Branch,File 0, 71:29 -> 71:30 = (#4 - #5), #5 [3,0,2]
// CHECK:  Branch,File 0, 71:34 -> 71:35 = (#2 - #3), #3 [2,0,0]

bool func_do_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  do {} while (a && b && c && d && e && f);
  return false;
}

// CHECK-LABEL:  Decision,File 0, 84:16 -> 84:42 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 84:16 -> 84:17 = #10, ((#0 + #1) - #10) [1,6,0]
// CHECK:  Branch,File 0, 84:21 -> 84:22 = #11, (#10 - #11) [6,5,0]
// CHECK:  Branch,File 0, 84:26 -> 84:27 = #9, (#8 - #9) [5,4,0]
// CHECK:  Branch,File 0, 84:31 -> 84:32 = #7, (#6 - #7) [4,3,0]
// CHECK:  Branch,File 0, 84:36 -> 84:37 = #5, (#4 - #5) [3,2,0]
// CHECK:  Branch,File 0, 84:41 -> 84:42 = #3, (#2 - #3) [2,0,0]

bool func_do_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  do {} while (a || b || c || d || e || f);
  return false;
}

// CHECK-LABEL:  Decision,File 0, 97:16 -> 97:42 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 97:16 -> 97:17 = ((#0 + #1) - #10), #10 [1,0,6]
// CHECK:  Branch,File 0, 97:21 -> 97:22 = (#10 - #11), #11 [6,0,5]
// CHECK:  Branch,File 0, 97:26 -> 97:27 = (#8 - #9), #9 [5,0,4]
// CHECK:  Branch,File 0, 97:31 -> 97:32 = (#6 - #7), #7 [4,0,3]
// CHECK:  Branch,File 0, 97:36 -> 97:37 = (#4 - #5), #5 [3,0,2]
// CHECK:  Branch,File 0, 97:41 -> 97:42 = (#2 - #3), #3 [2,0,0]

bool func_ternary_and(bool a, bool b, bool c, bool d, bool e, bool f) {
  return (a && b && c && d && e && f) ? true : false;
}

// CHECK-LABEL:  Decision,File 0, 110:11 -> 110:37 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 110:11 -> 110:12 = #10, (#0 - #10) [1,6,0]
// CHECK:  Branch,File 0, 110:16 -> 110:17 = #11, (#10 - #11) [6,5,0]
// CHECK:  Branch,File 0, 110:21 -> 110:22 = #9, (#8 - #9) [5,4,0]
// CHECK:  Branch,File 0, 110:26 -> 110:27 = #7, (#6 - #7) [4,3,0]
// CHECK:  Branch,File 0, 110:31 -> 110:32 = #5, (#4 - #5) [3,2,0]
// CHECK:  Branch,File 0, 110:36 -> 110:37 = #3, (#2 - #3) [2,0,0]

bool func_ternary_or(bool a, bool b, bool c, bool d, bool e, bool f) {
  return (a || b || c || d || e || f) ? true : false;
}

// CHECK-LABEL: Decision,File 0, 122:11 -> 122:37 = M:0, C:6
// CHECK-NEXT:  Branch,File 0, 122:11 -> 122:12 = (#0 - #10), #10 [1,0,6]
// CHECK:  Branch,File 0, 122:16 -> 122:17 = (#10 - #11), #11 [6,0,5]
// CHECK:  Branch,File 0, 122:21 -> 122:22 = (#8 - #9), #9 [5,0,4]
// CHECK:  Branch,File 0, 122:26 -> 122:27 = (#6 - #7), #7 [4,0,3]
// CHECK:  Branch,File 0, 122:31 -> 122:32 = (#4 - #5), #5 [3,0,2]
// CHECK:  Branch,File 0, 122:36 -> 122:37 = (#2 - #3), #3 [2,0,0]
