// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -Wexperimental-lifetime-safety -fexperimental-lifetime-safety-max-cfg-blocks=3 -Wno-dangling %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAILOUT
// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -Wexperimental-lifetime-safety -Wno-dangling %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOBAILOUT

struct MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View();
  void use() const;
};

class TriviallyDestructedClass {
  View a, b;
};

//===----------------------------------------------------------------------===//
// Basic Definite Use-After-Free (-W...permissive)
// These are cases where the pointer is guaranteed to be dangling at the use site.
//===----------------------------------------------------------------------===//

void single_block_cfg() {
  MyObj* p;
  {
    MyObj s;
    p = &s;     // CHECK-BAILOUT: warning: object whose reference is captured does not live long enough
  }             // CHECK-BAILOUT: note: destroyed here
  (void)*p;     // CHECK-BAILOUT: note: later used here
}

void multiple_block_cfg() {
  View v;
  int a = 10;
  MyObj safe;
  {
    if (a > 5) {
      MyObj s;
      v = s;    // CHECK-NOBAILOUT: warning: object whose reference is captured does not live long enough
    } else {
      v = safe;
    }     
  }             // CHECK-NOBAILOUT: note: destroyed here
  v.use();      // CHECK-NOBAILOUT: note: later used here
}
