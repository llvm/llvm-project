// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics
class A
{
public:
    A() {}

    template <class _F>
        explicit A(_F&& __f);

    A(A&&) {}
    A& operator=(A&&) {return *this;}
};

template <class T>
void f(T t)
{
  A a;
  a = f(t);
}
