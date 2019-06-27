typedef signed int NSInteger;

int capturedTypedef(NSInteger x) {
  return capturedTypedef(x);
}
// CHECK1: "static int extracted(NSInteger x) {\nreturn capturedTypedef(x);\n}\n\n" [[@LINE-3]]
// RUN: clang-refactor-test perform -action extract -selected=%s:4:10-4:28 %s -std=c++14 | FileCheck --check-prefix=CHECK1 %s

using NSUInteger = unsigned int;

int capturedUsing(NSUInteger x) {
  return capturedUsing(x);
}
// CHECK2: "static int extracted(NSUInteger x) {\nreturn capturedUsing(x);\n}\n\n" [[@LINE-3]]
// RUN: clang-refactor-test perform -action extract -selected=%s:12:10-12:26 %s -std=c++14 | FileCheck --check-prefix=CHECK2 %s
