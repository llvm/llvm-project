// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s
// expected-no-diagnostics

class Loc {
  int x;
};
class P1 {
public:
  Loc l;
  void setLoc(Loc L) {
    l = L;
  }
  
};
class P2 {
public:
  int m;
  int accessBase() {
    return m;
  }
};
class Derived: public P1, public P2 {
};
int radar13445834(Derived *Builder, Loc l) {
  Builder->setLoc(l);
  return Builder->accessBase();
  
}