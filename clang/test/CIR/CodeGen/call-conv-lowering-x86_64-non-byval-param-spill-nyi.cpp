// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir \
// RUN:     %s -o /dev/null -verify

// -O1 is load-bearing twice: Impl's key function is not defined here, so the
// vtable and its thunk are only emitted when optimizing, and CIRSimplify,
// which folds the read of the const slot, only runs then.

struct NonTrivial {
  ~NonTrivial();
};

struct HasVirtual {
  virtual void h();
};

struct Middle : HasVirtual {};

struct Renderer {
  virtual void draw(NonTrivial);
};

// Renderer sits at a non-zero offset in Impl, so overriding draw needs a
// this-adjusting thunk.
struct Impl : Middle, Renderer {
  // expected-error@+2 {{does not name the caller's storage}}
  // expected-note@+1 {{see current operation}}
  void draw(const NonTrivial);
};

void emit() { new Impl; }
