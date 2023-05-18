// RUN: not %clang_cc1 -fsyntax-only -std=c++11 %s 2>&1 | FileCheck %s

struct E {
  int num;
  const int Cnum = 0;
  mutable int Mnum;
  static int Snum;
  const static int CSnum;
};

struct D {
  E e;
  const E Ce;
  mutable E Me;
  static E Se;
  const static E CSe;
  E &getE() const;
  const E &getCE() const;
};

struct C {
  D d;
  const D Cd;
  mutable D Md;
  static D Sd;
  const static D CSd;
  D &getD() const;
  const D &getCD() const;
};

struct B {
  C c;
  const C Cc;
  mutable C Mc;
  static C Sc;
  const static C CSc;
  C &getC() const;
  static C &getSC();
  const C &getCC() const;
  static const C &getSCC();
};

struct A {
  B b;
  const B Cb;
  mutable B Mb;
  static B Sb;
  const static B CSb;
  B &getB() const;
  static B &getSB();
  const B &getCB() const;
  static const B &getSCB();
};

A& getA();

// Valid assignment
void test1(A a, const A Ca) {
  a.b.c.d.e.num = 5;
  a.b.c.d.e.Mnum = 5;
  Ca.b.c.d.e.Mnum = 5;
  a.b.c.d.e.Snum = 5;
  Ca.b.c.d.e.Snum = 5;
  Ca.b.c.Md.e.num = 5;
  Ca.Mb.Cc.d.e.Mnum = 5;
  Ca.Mb.getC().d.e.num = 5;
  Ca.getSB().c.d.e.num = 5;
  a.getSCB().c.d.Me.num = 5;
  Ca.Cb.Cc.Cd.Ce.Snum = 5;
  // CHECK-NOT: error:
  // CHECK-NOT: note:
}

// One note
void test2(A a, const A Ca) {
  Ca.b.c.d.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ca'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ca'
  // CHECK-NOT: note:

  a.Cb.c.d.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cb'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cb'
  // CHECK-NOT: note:

  a.b.c.Cd.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:

  a.b.c.d.e.CSnum = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'CSnum'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'CSnum'
  // CHECK-NOT: note:

  a.b.c.d.e.Cnum = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cnum'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cnum'
  // CHECK-NOT: note:

  a.getCB().c.d.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'getCB'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'getCB'
  // CHECK-NOT: note:

  a.getSCB().c.d.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'getSCB'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'getSCB'
  // CHECK-NOT: note:
}

// Two notes
void test3(A a, const A Ca) {

  a.getSCB().Cc.d.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cc'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cc'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'getSCB'
  // CHECK-NOT: note:

  Ca.b.c.Cd.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ca'
  // CHECK-NOT: note:

  a.getCB().c.Cd.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'getCB'
  // CHECK-NOT: note:

  a.b.getCC().d.e.Cnum = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cnum'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cnum'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'getCC'
  // CHECK-NOT: note:

  a.b.c.Cd.Ce.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:

  a.b.CSc.Cd.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'CSc'
  // CHECK-NOT: note:

  a.CSb.c.Cd.e.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Cd'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'CSb'
  // CHECK-NOT: note:
}

// No errors
void test4(const A Ca) {
  // Mutable cases
  Ca.Mb.c.d.e.num = 5;
  Ca.CSb.Mc.d.e.num = 5;
  Ca.getCB().Mc.d.e.num = 5;
  Ca.getSCB().Mc.d.e.num = 5;

  // Returning non-const reference
  Ca.getB().c.d.e.num = 5;
  Ca.CSb.getC().d.e.num = 5;
  Ca.getCB().getC().d.e.num = 5;
  Ca.getSCB().getC().d.e.num = 5;

  // Returning non-const reference
  Ca.getSB().c.d.e.num = 5;
  Ca.CSb.getSC().d.e.num = 5;
  Ca.getCB().getSC().d.e.num = 5;
  Ca.getSCB().getSC().d.e.num = 5;

  // Static member
  Ca.Sb.c.d.e.num = 5;
  Ca.CSb.Sc.d.e.num = 5;
  Ca.getCB().Sc.d.e.num = 5;
  Ca.getSCB().Sc.d.e.num = 5;

  // CHECK-NOT: error:
  // CHECK-NOT: note:
}

// Only display notes for relevant cases.
void test5(const A Ca) {
  Ca.Mb.c.d.Ce.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ce'
  // CHECK-NOT: note:

  Ca.getB().c.d.Ce.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ce'
  // CHECK-NOT: note:

  Ca.getSB().c.d.Ce.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ce'
  // CHECK-NOT: note:

  Ca.Sb.c.d.Ce.num = 5;
  // CHECK-NOT: error:
  // CHECK: error:{{.*}} 'Ce'
  // CHECK-NOT: note:
  // CHECK: note:{{.*}} 'Ce'
  // CHECK-NOT: note:
}
