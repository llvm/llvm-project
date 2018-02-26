
struct AClass {
  void constMethod() const;
  void method();
};

struct AWrapper {
  const AClass &object(int x) const;
  AClass &object(int x);
};

void takesClass(AWrapper &ref) {
  int x = 0;
  ref.object(x).constMethod();
  int y = 0;
  ref.object(x).method();
}
// CHECK1: "AClass &object = ref.object(x);\nobject" [[@LINE-4]]:3 -> [[@LINE-4]]:16

// CHECK1-NEXT: "object" [[@LINE-4]]:3 -> [[@LINE-4]]:16

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:14:3 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:16:3 %s | FileCheck --check-prefix=CHECK1 %s

void variableNameSuggested(AWrapper &object) {
#ifdef IN_COMPOUND
  {
#endif
  object.object(21).constMethod();
  object.object(21).method();
#ifdef IN_COMPOUND
  }
#endif
}
// CHECK2: "AClass &object = object.object(21);\nobject" [[@LINE-6]]:3 -> [[@LINE-6]]:20

// CHECK2-NEXT: "object" [[@LINE-7]]:3 -> [[@LINE-7]]:20

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:29:3 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:29:3 %s -D IN_COMPOUND | FileCheck --check-prefix=CHECK2 %s

void takesClass2(AWrapper &ref) {
  int x = 0;
  if (x)
    ref.object(x).constMethod();
  ref.object(x).method();
}
// CHECK3: "AClass &object = ref.object(x);\n" [[@LINE-4]]:3 -> [[@LINE-4]]:3 [Symbol extracted-decl 0 1:9 -> 1:15]
// CHECK3-NEXT: "object" [[@LINE-4]]:5 -> [[@LINE-4]]:18 [Symbol extracted-decl-ref 0 1:1 -> 1:7]
// CHECK3-NEXT: "object" [[@LINE-4]]:3 -> [[@LINE-4]]:16 [Symbol extracted-decl-ref 0 1:1 -> 1:7]

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:45:5 -emit-associated %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:46:3 -emit-associated %s | FileCheck --check-prefix=CHECK3 %s

void takesClass4(AWrapper &ref) {
  int x = 0;
  if (x) {
    ref.object(x).constMethod();
    ref.object(x).method();
  }
}
// CHECK4: "AClass &object = ref.object(x);\nobject" [[@LINE-4]]:5 -> [[@LINE-4]]:18

// CHECK4-NEXT: "object" [[@LINE-5]]:5 -> [[@LINE-5]]:18

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:58:5 %s | FileCheck --check-prefix=CHECK4 %s

void insertIntoCommonCompound1(AWrapper &ref) {
#ifdef EMBED
  int x = 0;
  while (true) {
#endif
  int x = 0;
  if (x) {
    if (true) {
      int y = x;
      ref.object(x).constMethod();
    }
  } else {
    ref.object(x).method();
  }
// CHECK5: "AClass &object = ref.object(x);\n" [[@LINE-8]]:3 -> [[@LINE-8]]:3
// CHECK5-NEXT: "object" [[@LINE-6]]:7 -> [[@LINE-6]]:20
// CHECK5-NEXT: "object" [[@LINE-4]]:5 -> [[@LINE-4]]:18
#ifdef EMBED
  }
#endif
}
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:77:7 %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:80:5 %s -DEMBED | FileCheck --check-prefix=CHECK5 %s

void checkFirstStmtInCompoundPlacement(AWrapper &ref) {
  while (true) {
    ref.object(20);
    ref.object(20).method();
// CHECK6: "AClass &object = ref.object(20);\nobject" [[@LINE-2]]:5 -> [[@LINE-2]]:19
  }
}

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=%s:94:5 %s | FileCheck --check-prefix=CHECK6 %s

class ImplicitThisRewrite {
  AWrapper &ref;
  ImplicitThisRewrite(AWrapper &ref) : ref(ref) {}

  void method() {
    // implicit-this: +1:5  // IMPLICIT-THIS: "AClass &object = this->ref.object(1);\nobject" [[@LINE+1]]:5 -> [[@LINE+1]]:18
    ref.object(1).method(); // IMPLICIT-NO-THIS: "AClass &object = ref.object(1);\nobject" [[@LINE]]:5 -> [[@LINE]]:18
    ref.object(1).constMethod(); // IMPLICIT-THIS-ME: "object" [[@LINE]]:5 -> [[@LINE]]:18
    // implicit-this2: +1:5
    this->ref.object(1).method(); // IMPLICIT-THIS-MENEXT: "object" [[@LINE]]:5 -> [[@LINE]]:24
  }
};

// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=implicit-this %s | FileCheck --check-prefixes=IMPLICIT-NO-THIS,IMPLICIT-THIS-ME %s
// RUN: clang-refactor-test perform -action extract-repeated-expr-into-var -at=implicit-this2 %s | FileCheck --check-prefixes=IMPLICIT-THIS,IMPLICIT-THIS-ME %s
