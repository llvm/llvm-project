void disallowExtractOfSimpleExpressions(int param) {
  int var = param;
  short var2 = (short)(var);
  int x = 0;
  const char *s = "Test";
  char c = 'C';
  double y = ((2.1));
  const char *f = __func__;
}

// RUN: not clang-refactor-test initiate -action extract -selected=%s:2:13-2:18 -selected=%s:2:14-2:15 -selected=%s:2:16-2:18 %s 2>&1 | FileCheck %s
// RUN: not clang-refactor-test initiate -action extract -selected=%s:3:16-3:28 -selected=%s:4:11-4:12 -selected=%s:5:19-5:25 %s 2>&1 | FileCheck %s
// RUN: not clang-refactor-test initiate -action extract -selected=%s:6:12-6:15 -selected=%s:7:14-7:21 -selected=%s:8:19-8:27 %s 2>&1 | FileCheck %s

// CHECK: Failed to initiate the refactoring action (the selected expression is too simple)!

void allowOperationsOnSimpleExpression(int x, int y) {
  int z = x + y;
  int zz = 0 + 1;
  allowOperationsOnSimpleExpression(1, y);
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:18:11-18:16 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Initiated the 'extract' action at 18:11 -> 18:16

// RUN: clang-refactor-test initiate -action extract -selected=%s:19:12-19:17 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: Initiated the 'extract' action at 19:12 -> 19:17

// RUN: clang-refactor-test initiate -action extract -selected=%s:20:37-20:41 %s | FileCheck --check-prefix=CHECK3 %s
// CHECK3: Initiated the 'extract' action at 20:3 -> 20:42

void defaultParameter(int x = 0 + 1) {

}

struct Struct {
  int y = 22 + 21;

  Struct(int x) : y(x + 1) { }
};

int initializerExpression = 1 + 2;

// RUN: not clang-refactor-test initiate -action extract -selected=%s:32:31-32:36 -selected=%s:37:11-37:18 -selected=%s:39:21-39:26 -selected=%s:42:29-42:34 %s 2>&1 | FileCheck --check-prefix=NOT-IN-FUNC %s
// NOT-IN-FUNC: Failed to initiate the refactoring action (the selected expression is not in a function)!

void disallowWholeFunctionBody() {
  int x = 0;
}
// RUN: not clang-refactor-test initiate -action extract -selected=%s:47:34-49:2 %s 2>&1 | FileCheck --check-prefix=NOT-IN-FUNC %s
