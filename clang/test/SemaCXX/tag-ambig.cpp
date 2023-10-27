// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef struct Point Point;

namespace NameSpace {
  class Point;
}

using namespace NameSpace;

class Test
{
public:
  struct Point { };
  virtual bool testMethod (Test::Point& p) = 0;
};

// PR8151
namespace A { struct Face {}; }
namespace B { struct Face {}; }
using namespace A;
using namespace B;
 
class C {
  struct Face;
  Face *mFaces;
};
