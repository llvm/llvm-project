
struct AbstractClass {
  __attribute__((annotate("test")))
  virtual void pureMethod() = 0;
};

struct Target : AbstractClass {
};
// CHECK1: "void pureMethod() override;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-abstract-methods -at=%s:7:1 %s | FileCheck --check-prefix=CHECK1 %s
