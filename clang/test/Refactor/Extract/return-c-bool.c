#ifndef __cplusplus
#define bool _Bool
#define true 1
#define false 0
#endif
typedef struct {
  bool b;
} HasBool;

bool boolType(bool b, HasBool *s) {
  bool x = b && true;
  bool y = boolType(b, s);
  bool z = s->b;
  bool a = !b;
  return (b || true);
}
// RUN: clang-refactor-test perform -action extract -selected=%s:11:12-11:21 -selected=%s:12:12-12:26 --selected=%s:13:12-13:16 --selected=%s:14:12-14:14 --selected=%s:15:10-15:21 %s | FileCheck %s
// RUN: clang-refactor-test perform -action extract -selected=%s:11:12-11:21 -selected=%s:12:12-12:26 --selected=%s:13:12-13:16 --selected=%s:14:12-14:14 --selected=%s:15:10-15:21 %s -x c++ | FileCheck %s
// CHECK: "static bool extracted
;
int boolCompareOps(int x, int y) {
  bool a = x == y;
  bool b = x >= y;
  bool c = ((x < y));
  return 0;
}
// RUN: clang-refactor-test perform -action extract -selected=%s:22:12-22:18 -selected=%s:23:12-23:18 --selected=%s:24:12-24:21 %s | FileCheck %s
// RUN: clang-refactor-test perform -action extract -selected=%s:22:12-22:18 -selected=%s:23:12-23:18 --selected=%s:24:12-24:21 %s -x c++ | FileCheck %s
