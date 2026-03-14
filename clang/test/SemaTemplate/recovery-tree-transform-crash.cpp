// RUN: %clang_cc1 -verify -frecovery-ast -frecovery-ast-type %s

template <class T> struct Ptr { T *operator->() const; };

struct ABC {
  void run();
};

Ptr<ABC> call(int); // expected-note {{candidate function not viable}}

void test() {
  call()->run(undef); // expected-error {{no matching function for call to 'call'}} \
                         expected-error {{use of undeclared identifier}}
}
