
typedef struct {
  int width, height;
} Rectangle;

class PrivateInstanceVariables {
  int x;
  Rectangle r;

  int method() {
    int y = x;
    return r.width + r.height * x + y;
  }
// CHECK1: (int x) {\nint y = x;\nreturn y;\n}
// CHECK1-NEXT: extracted(x)
// CHECK1: extracted(const Rectangle &r, int x) {\nint y = x;\n    return r.width + r.height * x + y;\n}
// CHECK1-NEXT: extracted(r, x)
// CHECK1: extracted(const Rectangle &r, int x, int y) {\nreturn r.width + r.height * x + y;\n}
// CHECK1-NEXT: extracted(r, x, y)
};

// RUN: clang-refactor-test perform -action extract -selected=%s:11:5-11:14 -selected=%s:11:5-12:38 -selected=%s:12:12-12:38 %s | FileCheck --check-prefix=CHECK1 %s

class PrivateInstanceVariablesExplicitThis {
  int x;


  int method(int y) {
    y = this->x;
    {
      int x = (this)->x;
    }
    y = x;
    x = 0;
    this->x = 0;
    y = (y == 0 ? this : (PrivateInstanceVariablesExplicitThis *)0) -> x;
  }
// CHECK2: (const PrivateInstanceVariablesExplicitThis &object, int &y) {\ny = object.x;\n    {\n      int x = (object).x;\n    }\n    y = object.x;\n}
// CHECK2-NEXT: extracted(*this, y)
// CHECK2: extracted(const PrivateInstanceVariablesExplicitThis &object) {\nint x = (object).x;\n}
// CHECK2-NEXT: extracted(*this)
// CHECK2: (PrivateInstanceVariablesExplicitThis &object, int &y) {\ny = object.x;\n    {\n      int x = (object).x;\n    }\n    y = object.x;\n    object.x = 0;\n}
// CHECK2: (PrivateInstanceVariablesExplicitThis &object) {\nobject.x = 0;\n}
// CHECK2: (PrivateInstanceVariablesExplicitThis &object, int &y) {\ny = (y == 0 ? &object : (PrivateInstanceVariablesExplicitThis *)0) -> x;\n}

// RUN: clang-refactor-test perform -action extract -selected=%s:29:5-33:10 -selected=%s:31:7-31:24 -selected=%s:29:5-34:10 -selected=%s:35:5-35:16 -selected=%s:36:5-36:73 %s | FileCheck --check-prefix=CHECK2 %s
};

class PublicInstanceVariables { int private_;
public:
  int x;
  Rectangle r;

  void method() {
    int y; y = x;
    x = 0;
    int z = r.width + r.height * x + y;
    this->x = (this)->r.height + (y == 0 ? this : (PublicInstanceVariables *)0) -> x;
    x = private_;
  }
// CHECK3: (const PublicInstanceVariables &object, int &y) {\ny = object.x;\n}
// CHECK3: (PublicInstanceVariables &object, int &y) {\ny = object.x;\n    object.x = 0;\n}
// CHECK3: (PublicInstanceVariables &object) {\nobject.x = 0;\n}
// CHECK3: (const PublicInstanceVariables &object, int y) {\nint z = object.r.width + object.r.height * object.x + y;\n}
// CHECK3: (PublicInstanceVariables &object, int y) {\nobject.x = (object).r.height + (y == 0 ? &object : (PublicInstanceVariables *)0) -> x;\n}
// CHECK3: (PublicInstanceVariables &object, int private_) {\nobject.x = object.private_;\n}
  void constMethod() const {
    const_cast<PublicInstanceVariables *>(this)->x = 2;
  }
// CHECK3: (const PublicInstanceVariables &object) {\nconst_cast<PublicInstanceVariables *>(&object)->x = 2;\n}
};

// RUN: clang-refactor-test perform -action extract -selected=%s:55:12-55:17 -selected=%s:55:12-56:10 -selected=%s:56:5-56:10 -selected=%s:57:5-57:39 -selected=%s:58:5-58:85 -selected=%s:59:5-59:17 -selected=%s:68:5-68:55 %s | FileCheck --check-prefix=CHECK3 %s
