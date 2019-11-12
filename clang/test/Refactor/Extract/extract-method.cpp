
void function(int x) {
  int y = x * x;
}

class InitiateMethodExtraction {
public:

  InitiateMethodExtraction() {
    int x = constMethod(1);
  }
// CHECK1: "void extracted() {\nint x = constMethod(1);\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted();" [[@LINE-3]]:5 -> [[@LINE-3]]:28
// CHECK2: "static void extracted(const InitiateMethodExtraction &object) {\nint x = object.constMethod(1);\n}\n\n"
// CHECK2-NEXT: "extracted(*this);"
;
  ~InitiateMethodExtraction() {
    int x = constMethod(2);
  }
// CHECK1: "void extracted() {\nint x = constMethod(2);\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted();" [[@LINE-3]]:5 -> [[@LINE-3]]:28
// CHECK2: "static void extracted(const InitiateMethodExtraction &object) {\nint x = object.constMethod(2);\n}\n\n"
// CHECK2-NEXT: "extracted(*this);"
;
  void method() {
    int x = constMethod(3);
  }
// CHECK1: "void extracted() {\nint x = constMethod(3);\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted();" [[@LINE-3]]:5 -> [[@LINE-3]]:28
// CHECK2: "static void extracted(const InitiateMethodExtraction &object) {\nint x = object.constMethod(3);\n}\n\n"
// CHECK2-NEXT: "extracted(*this);"
;
  int constMethod(int x) const {
    return x + x * 2;
  }
// CHECK1: "int extracted(int x) const {\nreturn x + x * 2;\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted(x);" [[@LINE-3]]:5 -> [[@LINE-3]]:22
// CHECK2: "static int extracted(int x) {\nreturn x + x * 2;\n}\n\n"
// CHECK2-NEXT: "extracted(x);"
;
  int operator << (int x) {
    return constMethod(x);
  }
// CHECK1: "int extracted(int x) {\nreturn constMethod(x);\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted(x)" [[@LINE-3]]:12 -> [[@LINE-3]]:26
// CHECK2: "static int extracted(const InitiateMethodExtraction &object, int x) {\nreturn object.constMethod(x);\n}\n\n"
// CHECK2-NEXT: "extracted(*this, x)"
;
  static void staticMethod(int x) {
    int y = x * x;
  }
// CHECK1: "static void extracted(int x) {\nint y = x * x;\n}\n\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK1-NEXT: "extracted(x);" [[@LINE-3]]:5 -> [[@LINE-3]]:19
// CHECK2: "static void extracted(int x) {\nint y = x * x;\n}\n\n"
// CHECK2-NEXT: "extracted(x);"
;
  void otherMethod();
};

void InitiateMethodExtraction::otherMethod() {
  int x = constMethod(4);
}
// CHECK1: "void extracted();\n\n" [[@LINE-6]]:3 -> [[@LINE-6]]:3
// CHECK1-NEXT: "void InitiateMethodExtraction::extracted() {\nint x = constMethod(4);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK1-NEXT: "extracted();" [[@LINE-4]]:3 -> [[@LINE-4]]:26
// CHECK2: "static void extracted(const InitiateMethodExtraction &object) {\nint x = object.constMethod(4);\n}\n\n"
// CHECK2-NEXT: "extracted(*this);"

// RUN: clang-refactor-test list-actions -at=%s:10:5 -selected=%s:10:5-10:27 %s | FileCheck --check-prefix=CHECK-METHOD %s

// CHECK-METHOD:      Extract Function{{$}}
// CHECK-METHOD-NEXT: Extract Method{{$}}

// RUN: clang-refactor-test list-actions -at=%s:3:3 -selected=%s:3:3-3:16 %s | FileCheck --check-prefix=CHECK-FUNC %s

// CHECK-FUNC:     Extract Function{{$}}
// CHECK-FUNC-NOT: Extract Method

// RUN: clang-refactor-test perform -action extract-method -selected=%s:10:5-10:27 -selected=%s:18:5-18:27 -selected=%s:26:5-26:27 -selected=%s:34:5-34:21 -selected=%s:42:12-42:26 -selected=%s:50:5-50:18 -selected=%s:61:3-61:25 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:10:5-10:27 -selected=%s:18:5-18:27 -selected=%s:26:5-26:27 -selected=%s:34:5-34:21 -selected=%s:42:12-42:26 -selected=%s:50:5-50:18 -selected=%s:61:3-61:25 %s | FileCheck --check-prefix=CHECK2 %s







namespace ns {
  struct Outer {
    struct Inner {
      Inner(int x);

      // comment
      void method2(int x) const; // this stays!
      int field;
    };
  };
}
ns::Outer::Inner::Inner(int x) {
  int y = x + x;
}
// CHECK3: "void extracted(int x);\n\n" [[@LINE-11]]:7 -> [[@LINE-11]]:7
// CHECK3: "void ns::Outer::Inner::extracted(int x) {\nint y = x + x;\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK3: "extracted(x);" [[@LINE-4]]:3 -> [[@LINE-4]]:17

namespace ns {

  // comment
  void Outer::Inner::method2(int x) const {
    if (x != 0)
      int z = x * x;
    else {
      (void)x;
    }
  }
// CHECK3: "void extracted(int x) const;\n\n" [[@LINE-23]]:7 -> [[@LINE-23]]:7
// CHECK3: "void Outer::Inner::extracted(int x) const {\nif (x != 0)\n      int z = x * x;\n    else {\n      (void)x;\n    }\n}\n\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3
// CHECK3: "extracted(x);" [[@LINE-8]]:5 -> [[@LINE-4]]:6
;
  void Outer::Inner::methodNotDeclared(int x) {
    int z = x * x;
  }
}
// CHECK3: "\n\nvoid extracted(int x);\n" [[@LINE-30]]:48 -> [[@LINE-30]]:48
// CHECK3: "void Outer::Inner::extracted(int x) {\nint z = x * x;\n}\n\n" [[@LINE-5]]:3 -> [[@LINE-5]]:3
// CHECK3: "extracted(x);" [[@LINE-5]]:5 -> [[@LINE-5]]:19

struct Empty {
  int field;
};

void Empty::method() {
  field = 22;
}
// CHECK3: "void extracted();\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK3: "void Empty::extracted() {\nfield = 22;\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK3: "extracted()" [[@LINE-4]]:3 -> [[@LINE-4]]:13

// RUN: clang-refactor-test perform -action extract-method -selected=%s:100:3-100:16 -selected=%s:110:5-114:4 -selected=%s:121:5-121:18 -selected=%s:133:3-133:13 %s | FileCheck --check-prefix=CHECK3 %s

template<typename T1, typename T2, int X>
struct TemplateExtraction {
  void method(); // CHECK4: "void extracted();\n\n" [[@LINE]]:3 -> [[@LINE]]:3
};

template<typename T1, typename T2, int x> // CHECK4: "template <typename T1, typename T2, int x> \nvoid TemplateExtraction<T1, T2, x>::extracted() {\nint y = x;\n}\n\n" [[@LINE]]:1 -> [[@LINE]]:1
void TemplateExtraction<T1, T2, x>::method() {
// template-method-begin: +1:1
  int y = x;
// template-method-end: +0:1
}

// RUN: clang-refactor-test perform -action extract-method -selected=template-method %s | FileCheck --check-prefix=CHECK4 %s

// UNSUPPORTED: system-windows
