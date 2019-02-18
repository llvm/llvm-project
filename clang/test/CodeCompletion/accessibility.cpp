class X {
public:
 int pub;
protected:
 int prot;
private:
 int priv;
};

class Unrelated {
public:
  static int pub;
protected:
  static int prot;
private:
  static int priv;
};

class Y : public X {
  int test() {
    this->pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:21:11 %s -o - \
    // RUN: | FileCheck -check-prefix=THIS %s
    // THIS: priv (InBase,Inaccessible)
    // THIS: prot (InBase)
    // THIS: pub (InBase)
    //
    // Also check implicit 'this->', i.e. complete at the start of the line.
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:21:1 %s -o - \
    // RUN: | FileCheck -check-prefix=THIS %s

    X().pub + 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:32:9 %s -o - \
    // RUN: | FileCheck -check-prefix=X-OBJ %s
    // X-OBJ: priv (Inaccessible)
    // X-OBJ: prot (Inaccessible)
    // X-OBJ: pub : [#int#]pub
    
    Y().pub + 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:39:9 %s -o - \
    // RUN: | FileCheck -check-prefix=Y-OBJ %s
    // Y-OBJ: priv (InBase,Inaccessible)
    // Y-OBJ: prot (InBase)
    // Y-OBJ: pub (InBase)

    this->X::pub = 10;
    X::pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:46:14 %s -o - \
    // RUN: | FileCheck -check-prefix=THIS-BASE %s
    //
    // THIS-BASE: priv (Inaccessible)
    // THIS-BASE: prot : [#int#]prot
    // THIS-BASE: pub : [#int#]pub
    //
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:47:8 %s -o - \
    // RUN: | FileCheck -check-prefix=THIS-BASE %s
    

    this->Unrelated::pub = 10; // a check we don't crash in this cases.
    Y().Unrelated::pub = 10; // a check we don't crash in this cases.
    Unrelated::pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:22 %s -o - \
    // RUN: | FileCheck -check-prefix=UNRELATED %s
    // UNRELATED: priv (Inaccessible)
    // UNRELATED: prot (Inaccessible)
    // UNRELATED: pub : [#int#]pub
    //
    // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:60:20 %s -o - \
    // RUN: | FileCheck -check-prefix=UNRELATED %s
    // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:61:16 %s -o - \
    // RUN: | FileCheck -check-prefix=UNRELATED %s
  }
};

class Outer {
 public:
  static int pub;
 protected:
  static int prot;
 private:
  static int priv;

  class Inner {
    int test() {
      Outer::pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:85:14 %s -o - \
    // RUN: | FileCheck -check-prefix=OUTER %s
    // OUTER: priv : [#int#]priv
    // OUTER: prot : [#int#]prot
    // OUTER: pub : [#int#]pub

    // Also check the unqualified case.
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:85:1 %s -o - \
    // RUN: | FileCheck -check-prefix=OUTER %s
    }
  };
};

class Base {
public:
  int pub;
};

class Accessible : public Base {
};

class Inaccessible : private Base {
};

class Test : public Accessible, public Inaccessible {
  int test() {
    this->Accessible::pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:112:23 %s -o - \
    // RUN: | FileCheck -check-prefix=ACCESSIBLE %s
    // ACCESSIBLE: pub (InBase)

    this->Inaccessible::pub = 10;
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:117:25 %s -o - \
    // RUN: | FileCheck -check-prefix=INACCESSIBLE %s
    // INACCESSIBLE: pub (InBase,Inaccessible)
  }
};
