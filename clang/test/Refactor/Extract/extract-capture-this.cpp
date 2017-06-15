
class AClassWithMethods {
  void method() {
  }
  void constMethod() const {
  }
  int operator << (int x) const { return x; }
  int operator >> (int x) { return x; }

  void toBeExtracted() {
    method();
    constMethod();
    *this << 2;
    *(this) >> 2;
    this->constMethod();
    (*(this)).constMethod();
  }
// CHECK1: (AClassWithMethods &object) {\nobject.method();\n}
// CHECK1: (const AClassWithMethods &object) {\nobject.constMethod();\n}
// CHECK1: (AClassWithMethods &object) {\nobject.method();\n    object.constMethod();\n}
// CHECK1: (const AClassWithMethods &object) {\nreturn object << 2;\n}
// CHECK1: (AClassWithMethods &object) {\nreturn (object) >> 2;\n}
// CHECK1: (const AClassWithMethods &object) {\nobject.constMethod();\n}
// CHECK1: (const AClassWithMethods &object) {\n((object)).constMethod();\n}

  void toBeExtracted2();
};
// RUN: clang-refactor-test perform -action extract -selected=%s:11:5-11:13 -selected=%s:12:5-12:17 -selected=%s:11:5-12:17 -selected=%s:13:5-13:14 -selected=%s:14:5-14:16 -selected=%s:15:5-15:24 -selected=%s:16:5-16:28 %s | FileCheck --check-prefix=CHECK1 %s

void takesRef(AClassWithMethods &object) {}
void takesConstRef(const AClassWithMethods &object) {}
void takesPtr(AClassWithMethods *object) {}
void takesConstPtr(const AClassWithMethods *object) {}

void AClassWithMethods::toBeExtracted2() {
  takesRef(*this);
  takesConstRef((*(this)));
  takesPtr(this);
  takesConstPtr((this));
  takesConstPtr(false ? this : (AClassWithMethods*)0);
}
// CHECK2: (AClassWithMethods &object) {\ntakesRef(object);\n}
// CHECK2: (const AClassWithMethods &object) {\ntakesConstRef(((object)));\n}
// CHECK2: (AClassWithMethods &object) {\ntakesPtr(&object);\n}
// CHECK2: (const AClassWithMethods &object) {\ntakesConstPtr((&object));\n}
// CHECK2: (AClassWithMethods &object) {\ntakesConstPtr(false ? &object : (AClassWithMethods*)0);\n}

// RUN: clang-refactor-test perform -action extract -selected=%s:36:3-36:18 -selected=%s:37:3-37:27 -selected=%s:38:3-38:17 -selected=%s:39:3-39:24 -selected=%s:40:3-40:54 %s | FileCheck --check-prefix=CHECK2 %s

#ifdef USECONST
#define CONST const
#else
#define CONST
#endif

class FallbackToMethodConstness {
  int getter() const { return 0; }
  int method(int x, FallbackToMethodConstness *other) CONST {
    return (x == 0 ? this : other)->getter();
  }
// CHECK3: (FallbackToMethodConstness &object, FallbackToMethodConstness *other, int x) {\nreturn (x == 0 ? &object : other)->getter();\n}
// CHECK3-CONST: (const FallbackToMethodConstness &object, FallbackToMethodConstness *other, int x) {\nreturn (x == 0 ? &object : other)->getter();\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:59:5-59:45 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:59:5-59:45 %s -DUSECONST | FileCheck --check-prefix=CHECK3-CONST %s
