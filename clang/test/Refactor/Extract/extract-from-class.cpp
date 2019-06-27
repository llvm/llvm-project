
#ifdef MULTIPLE
class OuterClass {
#endif

class AClass {

  int method(int x) {
    return x + x * 2;
  }
// CHECK1: "static int extracted(int x) {\nreturn x + x * 2;\n}\n\n" [[@LINE-5]]:1
// CHECK2: "static int extracted(int x) {\nreturn x + x * 2;\n}\n\n" [[@LINE-9]]:1
;
  AClass(int x) {
    int y = x * 1;
  }
// CHECK1: "static int extracted(int x) {\nreturn x * 1;\n}\n\n" [[@LINE-11]]:1
// CHECK2: "static int extracted(int x) {\nreturn x * 1;\n}\n\n" [[@LINE-15]]:1
  ~AClass() {
    int x = 0 + 4;
  }
// CHECK1: "static int extracted() {\nreturn 0 + 4;\n}\n\n" [[@LINE-16]]:1
// CHECK2: "static int extracted() {\nreturn 0 + 4;\n}\n\n" [[@LINE-20]]:1
;
  int operator +(int x) {
    return x + x * 3;
  }
// CHECK1: "static int extracted(int x) {\nreturn x + x * 3;\n}\n\n" [[@LINE-22]]:1
// CHECK2: "static int extracted(int x) {\nreturn x + x * 3;\n}\n\n" [[@LINE-26]]:1
;
  void otherMethod(int x);
};

#ifdef MULTIPLE
}
#endif

// RUN: clang-refactor-test perform -action extract -selected=%s:9:12-9:21 -selected=%s:15:13-15:18 -selected=%s:20:13-20:18 -selected=%s:26:12-26:21 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:9:12-9:21 -selected=%s:15:13-15:18 -selected=%s:20:13-20:18 -selected=%s:26:12-26:21 %s -DMULTIPLE | FileCheck --check-prefix=CHECK2 %s
;
void AClass::otherMethod(int x) {
  int y = x * x + 10;
}
// CHECK3: "static int extracted(int x) {\nreturn x * x + 10;\n}\n\n" [[@LINE-3]]:1
// RUN: clang-refactor-test perform -action extract -selected=%s:42:11-42:21 %s | FileCheck --check-prefix=CHECK3 %s
