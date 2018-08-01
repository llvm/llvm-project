
struct AbstractClass {
  virtual void pureMethod() = 0;
};

#ifdef HAS_BODY
  #define BODY { }
#else
  #define BODY ;
#endif

struct Target1 : AbstractClass {
  void method1() BODY
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1
// CHECK2: "void pureMethod() override { \n  <#code#>\n}\n\n" [[@LINE-2]]:1

struct Target2 : AbstractClass {
  void method1() BODY
  void method2() BODY
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1
// CHECK2: "void pureMethod() override { \n  <#code#>\n}\n\n" [[@LINE-2]]:1

struct Target2_1 : AbstractClass {
  void method1() BODY
  void method2();
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1
// CHECK2: "void pureMethod() override;\n\n" [[@LINE-2]]:1

struct Target3 : AbstractClass {
  void method1() BODY
  void method2() BODY
  void method3() BODY
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1
// CHECK2: "void pureMethod() override { \n  <#code#>\n}\n\n" [[@LINE-2]]:1

struct Target4 : AbstractClass {
  void method1() BODY
  void method2() BODY
  void method3() BODY
  void method4() BODY
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1
// CHECK2: "void pureMethod() override { \n  <#code#>\n}\n\n" [[@LINE-2]]:1

// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:12:1 -at=%s:18:1 -at=%s:25:1 -at=%s:32:1 -at=%s:40:1 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:12:1 -at=%s:18:1 -at=%s:25:1 -at=%s:32:1 -at=%s:40:1 %s -DHAS_BODY | FileCheck --check-prefix=CHECK2 %s
