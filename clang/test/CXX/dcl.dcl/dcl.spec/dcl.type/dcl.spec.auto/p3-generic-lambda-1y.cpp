// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1y -DCXX1Y

//FIXME: These tests were written when return type deduction had not been implemented
// for generic lambdas, hence
template<class T> T id(T t);
template<class ... Ts> int vfoo(Ts&& ... ts);
auto GL1 = [](auto a, int i) -> int { return id(a); };

auto GL2 = [](auto ... As) -> int { return vfoo(As...); };
auto GL3 = [](int i, char c, auto* ... As) -> int { return vfoo(As...); };

auto GL4 = [](int i, char c, auto* ... As) -> int { return vfoo(As...); };


void foo() {
  auto GL1 = [](auto a, int i) -> int { return id(a); };

  auto GL2 = [](auto ... As) -> int { return vfoo(As...); };
}

int main()
{
  auto l1 = [](auto a) -> int { return a + 5; };
  auto l2 = [](auto *p) -> int { return p + 5; };
 
  struct A { int i; char f(int) { return 'c'; } };
  auto l3 = [](auto &&ur, 
                auto &lr, 
                auto v, 
                int i, 
                auto* p,
                auto A::*memvar,
                auto (A::*memfun)(int),
                char c,
                decltype (v)* pv
                , auto (&array)[5] 
              ) -> int { return v + i + c
                          + array[0]; 
                       };
  int arr[5] = {0, 1, 2, 3, 4 };
  int lval = 0;
  double d = 3.14;
  l3(3, lval, d, lval, &lval, &A::i, &A::f, 'c', &d, arr);
  auto l4 = [](decltype(auto) a) -> int { return 0; }; //expected-error{{decltype(auto)}}
  {
    struct Local {
      static int ifi(int i) { return i; }
      static char cfi(int) { return 'a'; }
      static double dfi(int i) { return i + 3.14; }
      static Local localfi(int) { return Local{}; }
    };
    auto l4 = [](auto (*fp)(int)) -> int { return fp(3); }; //expected-error{{no viable conversion from returned value of type 'Local' to function return type 'int'}} 
    l4(&Local::ifi);
    l4(&Local::cfi);
    l4(&Local::dfi);
    l4(&Local::localfi); //expected-note{{in instantiation of function template specialization}}  
  }
  {
    auto unnamed_parameter = [](auto, auto) -> void { };
    unnamed_parameter(3, '4');
  }
  {
    auto l = [](auto 
                      (*)(auto)) { }; //expected-error{{'auto' not allowed}}
    //FIXME: These diagnostics might need some work.
    auto l2 = [](char auto::*pm) { };  //expected-error{{cannot combine with previous}}\
                                         expected-error{{'pm' does not point into a class}}
    auto l3 = [](char (auto::*pmf)()) { };  //expected-error{{'auto' not allowed}}\
                                              expected-error{{'pmf' does not point into a class}}\
                                              expected-error{{function cannot return function type 'char ()'}}
  }
}


