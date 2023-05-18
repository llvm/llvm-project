// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s

class S {
 public:
  int a_;
  void s(int a) {
    a_ = a_;  // expected-warning {{assigning field to itself}}

    // Don't really care about this one either way.
    this->a_ = a_;  // expected-warning {{assigning field to itself}}

    a_ += a_;  // Shouldn't warn.
  }
};

void f0(S* s) {
  // Would be nice to have, but not important.
  s->a_ = s->a_;
}

void f1(S* s, S* t) {
  // Shouldn't warn.
  t->a_ = s->a_;
}

struct T {
  S* s_;
};

void f2(T* t) {
  // Would be nice to have, but even less important.
  t->s_->a_ = t->s_->a_;
}

void f3(T* t, T* t2) {
  // Shouldn't warn.
  t2->s_->a_ = t->s_->a_;
}

void f4(int i) {
  // This is a common pattern to silence "parameter unused". Shouldn't warn.
  i = i;

  int j = 0;
  j = j;  // Likewise.
}

@interface I {
  int a_;
}
@end

@implementation I
- (void)setA:(int)a {
  a_ = a_;  // expected-warning {{assigning instance variable to itself}}
}

- (void)foo:(I*)i {
  // Don't care much about this warning.
  i->a_ = i->a_;  // expected-warning {{assigning instance variable to itself}}

  // Shouldn't warn.
  a_ = i->a_;
  i->a_ = a_;
}
@end
