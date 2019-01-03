class X {
public:
 int pub;
protected:
 int prot;
private:
 int priv;
};

class Y : public X {
  int test() {
    []() {

      // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:13:1 %s -o - \
      // RUN: | FileCheck %s
      // CHECK: priv (InBase,Inaccessible)
      // CHECK: prot (InBase)
      // CHECK: pub (InBase)
    };
  }
};


