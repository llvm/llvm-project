// RUN: c-index-test -test-load-source all %s | FileCheck %s

class Test {
public:
  __attribute__((annotate("spiffy_method"))) void aMethod();

public __attribute__((annotate("works"))):
  void anotherMethod(); // annotation attribute should be propagated.

private __attribute__((annotate("investigations"))):
  //propagated annotation should have changed from "works" to "investigations"
  void inlineMethod() {}

protected:
  // attribute propagation should have stopped here
  void methodWithoutAttribute();
};

// CHECK: ClassDecl=Test:3:7 (Definition) Extent=[3:1 - 17:2]
// CHECK-NEXT: CXXAccessSpecifier=:4:1 (Definition) Extent=[4:1 - 4:8]
// CHECK-NEXT: CXXMethod=aMethod:5:51 Extent=[5:3 - 5:60]
// CHECK-NEXT: attribute(annotate)=spiffy_method Extent=[5:18 - 5:43]
// CHECK-NEXT: CXXAccessSpecifier=:7:1 (Definition) Extent=[7:1 - 7:43]
// CHECK-NEXT: attribute(annotate)=works Extent=[7:23 - 7:40]
// CHECK-NEXT: CXXMethod=anotherMethod:8:8 Extent=[8:3 - 8:23]
// CHECK-NEXT: attribute(annotate)=works Extent=[7:23 - 7:40]
// CHECK-NEXT: CXXAccessSpecifier=:10:1 (Definition) Extent=[10:1 - 10:53]
// CHECK-NEXT: attribute(annotate)=investigations Extent=[10:24 - 10:50]
// CHECK-NEXT: CXXMethod=inlineMethod:12:8 (Definition) Extent=[12:3 - 12:25]
// CHECK-NEXT: attribute(annotate)=investigations Extent=[10:24 - 10:50]
// CHECK-NEXT: CompoundStmt= Extent=[12:23 - 12:25]
// CHECK-NEXT: CXXAccessSpecifier=:14:1 (Definition) Extent=[14:1 - 14:11]
// CHECK-NEXT: CXXMethod=methodWithoutAttribute:16:8 Extent=[16:3 - 16:32]
