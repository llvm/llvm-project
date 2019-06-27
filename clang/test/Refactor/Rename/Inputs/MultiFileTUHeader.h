class Foo {  // CHECK: rename "{{.*}}/Inputs/MultiFileTUHeader.h" [[@LINE]]:7 -> [[@LINE]]:10
public:
  Foo();     // CHECK: rename "{{.*}}/Inputs/MultiFileTUHeader.h" [[@LINE]]:3 -> [[@LINE]]:6

  void method();
};
