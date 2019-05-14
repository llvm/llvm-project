// Merge success
namespace N1 {
  int x;
}

// Merge multiple namespaces
namespace N2 {
  extern int x;
}
namespace N2 {
  extern float y;
}

// Merge namespace with conflict
namespace N3 {
  extern float z;
}

namespace AliasWithSameName = N3;

namespace TestUnresolvedTypenameAndValueDecls {
template <class T> class Base {
public:
  typedef T foo;
  void bar();
};
}
