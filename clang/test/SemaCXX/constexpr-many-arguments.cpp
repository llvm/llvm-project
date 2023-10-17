// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fexperimental-new-constant-interpreter %s
// PR13197

struct type1
{
  constexpr type1(int a0) : my_data{a0} {}
  int my_data[1];
};

struct type2
{
  typedef type1 T;
  constexpr type2(T a00, T a01, T a02, T a03, T a04, T a05, T a06, T a07, T a08, T a09,
                       T a10, T a11, T a12, T a13, T a14, T a15, T a16, T a17, T a18, T a19,
                       T a20, T a21, T a22) 
    : my_data{a00, a01, a02, a03, a04, a05, a06, a07, a08, a09,
              a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
              a20, a21, a22}
  {}
  type1 my_data[23];
};

struct type3
{
  constexpr type3(type2 a0, type2 a1) : my_data{a0, a1} {}
  type2 my_data[2];
};

constexpr type3 g
{
  {
   {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},
   {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},
   {0},{0},{0}
  }, 
  {
   {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},
   {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},
   {0},{0},{0}
  }
};

