#ifndef REDUNDANT_OVERRIDE_HEADER_H
#define REDUNDANT_OVERRIDE_HEADER_H

struct HeaderBase {
  virtual void f() {}
  virtual int value(int n) const { return n; }
};

struct HeaderDerived : HeaderBase {
  void f() override;
  int value(int n) const override;
};
// CHECK-FIXES: struct HeaderDerived : HeaderBase {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES-NOT: value(
// CHECK-FIXES: };

struct HeaderVisibility : protected HeaderBase {
  void f() override;
};
// CHECK-FIXES: struct HeaderVisibility : protected HeaderBase {
// CHECK-FIXES-NEXT: void f() override;

#endif
