@import namespaces_top;

double &global(double);
double &global2(double);

namespace LookupBeforeImport {
  double &f(double);
}

namespace N2 { }

namespace N2 { }

namespace N2 { }

namespace N2 { }

namespace N2 { 
  double& f(double);
}

namespace N3 { 
  double& f(double);
}

namespace N5 {
  double &f(double);
}

namespace N6 {
  double &f(double);
}

namespace N7 {
  double &f(double);
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
  void consumeFoo(Foo*);
}

namespace N12 {
  namespace {
    class Foo;
  }
  void consumeFoo(Foo*);
}

namespace Empty {}
