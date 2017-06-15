
void function(int x) {
  int y = x * x;
}

@interface MethodExtraction

- (int)method;
- (void)classMethod;

@end

@implementation MethodExtraction

- (void)aMethod:(int)x withY:(int)y {
  int a = x + y;
}
// CHECK1: "- (void)extracted:(int)x y:(int)y {\nint a = x + y;\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK1-NEXT: "[self extracted:x y:y];" [[@LINE-3]]:3 -> [[@LINE-3]]:17
// CHECK2: "static void extracted(int x, int y) {\nint a = x + y;\n}\n\n"
// CHECK2-NEXT: "extracted(x, y);"
;
+ (void)classMethod {
  int x = 1;
  int y = function(x);
}
// CHECK1: "+ (void)extracted {\nint x = 1;\n  int y = function(x);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK1-NEXT: "[self extracted];" [[@LINE-4]]:3 -> [[@LINE-3]]:23
// CHECK2: "static void extracted() {\nint x = 1;\n  int y = function(x);\n}\n\n"
// CHECK2-NEXT: "extracted();"

@end

@implementation MethodExtraction (Category)

- (void)catMethod {
  int x = [self method];
}

+ (void)catClassMethod {
  int x = function(42);
}

@end

// RUN: clang-refactor-test list-actions -at=%s:16:3 -selected=%s:16:3-16:16 %s | FileCheck --check-prefix=CHECK-METHOD %s

// CHECK-METHOD:      Extract Function{{$}}
// CHECK-METHOD-NEXT: Extract Method{{$}}

// RUN: clang-refactor-test list-actions -at=%s:3:3 -selected=%s:3:3-3:16 %s | FileCheck --check-prefix=CHECK-FUNC %s

// CHECK-FUNC:     Extract Function{{$}}
// CHECK-FUNC-NOT: Extract Method

// RUN: clang-refactor-test perform -action extract-method -selected=%s:16:3-16:16 -selected=%s:24:3-25:22 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:16:3-16:16 -selected=%s:24:3-25:22 %s | FileCheck --check-prefix=CHECK2 %s
