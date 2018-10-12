#include <functional>

int foo(int x, int y) {
  return x + y - 1; // Source foo start line
}

struct Bar {
   int operator()() {
       return 66 ; // Source Bar::operator()() start line
   }
   int add_num(int i) const { return i + 3 ; } // Source Bar::add_num start line
   int num_ = 0 ;
} ;

int main (int argc, char *argv[])
{
  int acc = 42;
  std::function<int (int,int)> f1 = foo;
  std::function<int (int)> f2 = [acc,f1] (int x) -> int {
    return x+f1(acc,x); // Source lambda used by f2 start line
  };

  auto f = [](int x, int y) { return x + y; }; // Source lambda used by f3 start line
  auto g = [](int x, int y) { return x * y; } ;
  std::function<int (int,int)> f3 =  argc %2 ? f : g ;

  Bar bar1 ;
  std::function<int ()> f4( bar1 ) ;
  std::function<int (const Bar&, int)> f5 = &Bar::add_num;
  std::function<int(Bar const&)> f_mem = &Bar::num_;

  return f_mem(bar1) +     // Set break point at this line.
         f1(acc,acc) +     // Source main invoking f1
         f2(acc) +         // Set break point at this line.
         f3(acc+1,acc+2) + // Set break point at this line. 
         f4() +            // Set break point at this line. 
         f5(bar1, 10);     // Set break point at this line.
}
