// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s -DDELAYED_TEMPLATE_PARSING
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fms-extensions %s -DMS_EXTENSIONS
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fdelayed-template-parsing -fms-extensions %s -DMS_EXTENSIONS -DDELAYED_TEMPLATE_PARSING


namespace test_star_this {
namespace ns1 {
class A {
  int x = 345;
  auto foo() {
    (void) [*this, this] { };  //expected-error{{'this' can appear only once}}
    (void) [this] { ++x; };
    (void) [*this] { ++x; };  //expected-error{{read-only variable}}
    (void) [*this] () mutable { ++x; };
    (void) [=] { return x; };
    (void) [&, this] { return x; };
    (void) [=, *this] { return x; };
    (void) [&, *this] { return x; };
  }
};
} // end ns1

namespace ns2 {
  class B {
    B(const B&) = delete; //expected-note{{deleted here}}
    int *x = (int *) 456;
    void foo() {
      (void)[this] { return x; };
      (void)[*this] { return x; }; //expected-error{{call to deleted}}
    }
  };
} // end ns2
namespace ns3 {
  class B {
    B(const B&) = delete; //expected-note2{{deleted here}}
    
    int *x = (int *) 456;
    public: 
    template<class T = int>
    void foo() {
      (void)[this] { return x; };
      (void)[*this] { return x; }; //expected-error2{{call to deleted}}
    }
    
    B() = default;
  } b;
  B *c = (b.foo(), nullptr); //expected-note{{in instantiation}}
} // end ns3

namespace ns4 {
template<class U>
class B {
  B(const B&) = delete; //expected-note{{deleted here}}
  double d = 3.14;
  public: 
  template<class T = int>
  auto foo() {
    const auto &L = [*this] (auto a) mutable { //expected-error{{call to deleted}}
      d += a; 
      return [this] (auto b) { return d +=b; }; 
    }; 
  }
  
  B() = default;
};
void main() {
  B<int*> b;
  b.foo(); //expected-note{{in instantiation}}
} // end main  
} // end ns4
} //end ns test_star_this
