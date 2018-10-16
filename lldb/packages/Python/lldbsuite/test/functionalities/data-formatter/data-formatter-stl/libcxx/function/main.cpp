//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <functional>

int foo(int x, int y) {
  return x + y - 1;
}

struct Bar {
   int operator()() {
       return 66 ;
   }
   int add_num(int i) const { return i + 3 ; }
} ;

int main (int argc, char *argv[])
{
  int acc = 42;
  std::function<int (int,int)> f1 = foo;
  std::function<int (int)> f2 = [acc,f1] (int x) -> int {
    return x+f1(acc,x);
  };

  auto f = [](int x, int y) { return x + y; };
  auto g = [](int x, int y) { return x * y; } ;
  std::function<int (int,int)> f3 =  argc %2 ? f : g ;

  Bar bar1 ;
  std::function<int ()> f4( bar1 ) ;
  std::function<int (const Bar&, int)> f5 = &Bar::add_num;

  return f1(acc,acc) + f2(acc) + f3(acc+1,acc+2) + f4() + f5(bar1, 10); // Set break point at this line.
}
