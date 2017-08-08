@interface NSArray
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

void extractedStmtNoNeedForSemicolon(NSArray *array) {
  for (id i in array) {
    int x = 0;
  }
// CHECK1: "static void extracted(NSArray *array) {\nfor (id i in array) {\n    int x = 0;\n  }\n}\n\n"
  id lock;
  @synchronized(lock) {
    int x = 0;
  }
// CHECK1: "static void extracted(id lock) {\n@synchronized(lock) {\n    int x = 0;\n  }\n}\n\n"
  @autoreleasepool {
    int x = 0;
  }
// CHECK1: "static void extracted() {\n@autoreleasepool {\n    int x = 0;\n  }\n}\n\n"
  @try {
    int x = 0;
  } @finally {
  }
// CHECK1: "static void extracted() {\n@try {\n    int x = 0;\n  } @finally {\n  }\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:6:3-8:4 -selected=%s:11:3-13:4 -selected=%s:15:3-17:4 -selected=%s:19:3-22:4 %s | FileCheck --check-prefix=CHECK1 %s

@interface I

@end

@implementation I

- (int)inferReturnTypeFromReturnStatement:(int)x {
  if (x == 0) {
    return x;
  }
  if (x == 1) {
    return x + 1;
  }
  return x + 2;
}
// CHECK2: "static int extracted(int x) {\nif (x == 1) {\n    return x + 1;\n  }\n  return x + 2;\n}\n\n"
// CHECK2: "static int extracted(int x) {\nif (x == 1) {\n    return x + 1;\n  }\n}\n\n"

// RUN: clang-refactor-test perform -action extract -selected=%s:38:3-41:15 -selected=%s:38:3-40:4 %s | FileCheck --check-prefix=CHECK2 %s

@end

void partiallySelectedWithImpCastCrash(I *object) {
// partially-selected-begin: +1:3
  object;
// partially-selected-end: +1:11
// comment
// CHECK3: "static void extracted(I *object) {\nobject;\n}\n\n"
// RUN: clang-refactor-test perform -action extract -selected=partially-selected %s | FileCheck --check-prefix=CHECK3 %s
}
