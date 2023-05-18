namespace RedeclAcrossImport {
  enum E { e };
}

namespace AddAndReexportBeforeImport {
  struct S {};
  extern struct S t;
}

@import namespaces_top;

namespace RedeclAcrossImport {
  E x = e;
}

float &global(float);
float &global2(float);

namespace LookupBeforeImport {
  float &f(float);
}

namespace N1 { }

namespace N1 { 
  float& f(float);
}

namespace N2 { 
  float& f(float);
}





namespace N5 {
  int &f(int);
}

namespace N6 {
  int &f(int);
}

namespace N7 {
  int &f(int);
}

namespace N8 {
  int &f(int);
}

namespace N9 {
  int &f(int);
}

namespace N10 {
  int &f(int);
}

namespace N11 {
  namespace {
    class Foo;
  }
  Foo *getFoo();
}

namespace N12 {
  namespace {
    class Foo;
  }
  Foo *getFoo();
}

namespace Empty {}
