
// comment 1
void extractBeforeComment1(int x) {
  int y = x * x;
}
// CHECK1: "static int extracted(int x) {\nreturn x * x;\n}\n\n" [[@LINE-4]]:1

/* comment 2 */

void extractBeforeComment2(int x) {
  int y = x * x;
}
// CHECK1: "static int extracted(int x) {\nreturn x * x;\n}\n\n" [[@LINE-5]]:1

/// comment 1
///
/// line 2
void extractBeforeDocComment1(int x) {
  int y = x * x;
}
// CHECK1: "static int extracted(int x) {\nreturn x * x;\n}\n\n" [[@LINE-6]]:1

/**
 * @brief extractBeforeDocComment2
 * @param x
 */
void extractBeforeDocComment2(int x) {
  int y = x * x;
}
// CHECK1: "static int extracted(int x) {\nreturn x * x;\n}\n\n" [[@LINE-7]]:1

// RUN: clang-refactor-test perform -action extract -selected=%s:4:11-4:16 -selected=%s:11:11-11:16 -selected=%s:19:11-19:16 -selected=%s:28:11-28:16 %s | FileCheck --check-prefix=CHECK1 %s

/**
 * @brief The AClass class
 */
class AClass {

  /// doc comment
  int method(int x) {
    return x * x;
  }
// CHECK2: "static int extracted(int x) {\nreturn x * x;\n}\n\n" [[@LINE-9]]:1
};

// RUN: clang-refactor-test perform -action extract -selected=%s:41:12-41:17 %s | FileCheck --check-prefix=CHECK2 %s

namespace {

} // end anonymous namespace

void afterBraceAfterComment() { // CHECK3: "static void extracted() {\nint x = 0;\n}\n\n" [[@LINE]]:1
// after-brace-begin: +1:1
  int x = 0;
// after-brace-end: +0:1
} // another trailing
// This is valid CHECK3: "static void extracted() {\nint x = 0;\n}\n\n" [[@LINE]]:1
void inbetweenerTwoComments() {
// inbetween-begin: +1:1
  int x = 0;
// inbetween-end: +0:1
}

// RUN: clang-refactor-test perform -action extract -selected=after-brace -selected=inbetween %s | FileCheck --check-prefix=CHECK3 %s
