// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -lifetime-safety-max-cfg-blocks=3 -Wno-dangling -verify=bailout %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify=bailout -verify=nobailout %s

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
    p = &s;     // bailout-warning {{object whose reference is captured does not live long enough}}
  }             // bailout-note {{destroyed here}}
  (void)*p;     // bailout-note {{later used here}}
}

void multiple_block_cfg() {
  MyObj* p;
  int a = 10;
  MyObj safe;
  {
    if (a > 5) {
      MyObj s;
      p = &s;    // nobailout-warning {{object whose reference is captured does not live long enough}}
    } else {     // nobailout-note {{destroyed here}}
      p = &safe;
    }
  }
  p->use();      // nobailout-note {{later used here}}
}
