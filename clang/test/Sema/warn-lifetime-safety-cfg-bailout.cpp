// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -lifetime-safety-max-cfg-blocks=3 -Wno-dangling -verify=CHECK-BAILOUT %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify=CHECK-BAILOUT -verify=CHECK-NOBAILOUT %s

struct MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);
  void use() const;
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
    p = &s;     // CHECK-BAILOUT-warning {{object whose reference is captured does not live long enough}}
  }             // CHECK-BAILOUT-note {{destroyed here}}
  (void)*p;     // CHECK-BAILOUT-note {{later used here}}
}

void multiple_block_cfg() {
  MyObj* p;
  int a = 10;
  MyObj safe;
  {
    if (a > 5) {
      MyObj s;
      p = &s;    // CHECK-NOBAILOUT-warning {{object whose reference is captured does not live long enough}}
    } else {     // CHECK-NOBAILOUT-note {{destroyed here}}
      p = &safe;
    }     
  }             
  p->use();      // CHECK-NOBAILOUT-note {{later used here}}
}
