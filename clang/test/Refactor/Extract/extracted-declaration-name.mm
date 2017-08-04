
int compute(int n, int x, int y) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
// extract-func-begin: +1:12
    sum += (x - i) * (y + i);
// extract-func-end: -1:29
  }
  return sum;
}
// RUN: clang-refactor-test perform -action extract -emit-associated -selected=extract-func %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: "static int extracted(int i, int x, int y) {\nreturn (x - i) * (y + i);\n}\n\n" [[@LINE-10]]:1 -> [[@LINE-10]]:1 [Symbol extracted-decl 0 1:12 -> 1:21]
// CHECK1-NEXT: "extracted(i, x, y)" [[@LINE-7]]:12 -> [[@LINE-7]]:29 [Symbol extracted-decl-ref 0 1:1 -> 1:10]

struct Struct {
  int func(int y) { return y; }

  int compute(int x, int y);
};

int Struct::compute(int x, int y) {
// extract-member-func-begin: +1:10
  return x * func(y + x) + y;
// extract-member-func-end: -1:25
}
// RUN: clang-refactor-test perform -action extract-method -emit-associated -selected=extract-member-func %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: "int extracted(int x, int y);\n\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3 [Symbol extracted-decl 0 1:5 -> 1:14]
// CHECK2-NEXT: "int Struct::extracted(int x, int y) {\nreturn x * func(y + x);\n}\n\n" [[@LINE-7]]:1 -> [[@LINE-7]]:1 [Symbol extracted-decl 0 1:13 -> 1:22]
// CHECK2-NEXT: "extracted(x, y)" [[@LINE-6]]:10 -> [[@LINE-6]]:25 [Symbol extracted-decl-ref 0 1:1 -> 1:10]

@interface I

@property int p;

- (void)foo:(int)x with:(int)y;

@end

@implementation I

- (void)foo:(int)x with:(int)y {
// extract-selector-begin: +1:1
  int m = compute(10, self.p + y, x);
// extract-selector-end: +0:1
}

@end
// RUN: clang-refactor-test perform -action extract-method -emit-associated -selected=extract-selector %s | FileCheck --check-prefix=CHECK3 %s
// CHECK3: "- (void)extracted:(int)x y:(int)y {\nint m = compute(10, self.p + y, x);\n}\n\n" [[@LINE-8]]:1 -> [[@LINE-8]]:1 [Symbol extracted-decl 0 1:9 -> 1:18 1:26 -> 1:27]
// CHECK3-NEXT: "[self extracted:x y:y];" [[@LINE-7]]:3 -> [[@LINE-7]]:38 [Symbol extracted-decl-ref 0 1:7 -> 1:16 1:19 -> 1:20]
