// RUN: %check_clang_tidy "%s" readability-identifier-naming "%t"

template<class T>
struct X {
  struct B;
  struct A : public B {
    virtual void foo() { }
  };
};
