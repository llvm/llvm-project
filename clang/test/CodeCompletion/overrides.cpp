class A {
 public:
  virtual void vfunc(bool param);
  virtual void vfunc(bool param, int p);
  void func(bool param);
};
class B : public A {
virtual int ttt(bool param, int x = 3) const;
void vfunc(bool param, int p) override;
};
class C : public B {
 public:
  void vfunc(bool param) override;
  void
};

// Runs completion at ^void.
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:14:3 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: COMPLETION: Pattern : int ttt(bool param, int x = 3) const override{{$}}
// CHECK-CC1: COMPLETION: Pattern : void vfunc(bool param, int p) override{{$}}
// CHECK-CC1-NOT: COMPLETION: Pattern : void vfunc(bool param) override{{$}}
//
// Runs completion at vo^id.
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:14:5 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: COMPLETION: Pattern : void vfunc(bool param, int p) override{{$}}
// CHECK-CC2-NOT: COMPLETION: Pattern : int ttt(bool param, int x = 3) const override{{$}}
// CHECK-CC2-NOT: COMPLETION: Pattern : void vfunc(bool param) override{{$}}
//
// Runs completion at void ^.
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:14:8 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3-NOT: COMPLETION: Pattern : int ttt(bool param, int x = 3) const override{{$}}
// CHECK-CC3-NOT: COMPLETION: Pattern : void vfunc(bool param, int p) override{{$}}
// CHECK-CC3-NOT: COMPLETION: Pattern : void vfunc(bool param) override{{$}}
