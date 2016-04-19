#include "B.h"

template <class T>
class Expr {
public:
   void print(B::basic_ostream<char>& os) {
     os << B::setw(42);
     os << B::endl;
  }
};
