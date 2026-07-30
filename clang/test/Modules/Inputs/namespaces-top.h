namespace N1 { 
  int& f(int);
}

namespace N2 { 
  int& f(int);
}

namespace N3 { 
  int& f(int);
}

namespace N12 { }

namespace N13 {
  void f();
  int f(int);
  void (*p)() = &f;
}

namespace AddAndReexportBeforeImport {
  int S;
}

namespace Empty {}
