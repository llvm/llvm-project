// Header for PCH test namespaces.cpp

namespace N1 {
  typedef int t1;
}

namespace N1 {
  typedef int t2;

  void used_func();

  struct used_cls { };
}

namespace N2 {
  typedef float t1;

  namespace Inner {
    typedef int t3;
  };
}

namespace {
  void anon() { }
  class C;
}

namespace N3 {
  namespace {
    class C;
  }
}

namespace Alias1 = N2::Inner;

using namespace N2::Inner;

extern "C" {
  void ext();
}

inline namespace N4 { 
  struct MemberOfN4;
}
