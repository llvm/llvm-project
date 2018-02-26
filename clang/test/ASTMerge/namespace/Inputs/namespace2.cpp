// Merge success
namespace N1 {
  extern int x0;
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
  extern double z;
}

namespace Enclosing {
namespace Nested {
  const int z = 4;
}
}

namespace ContainsInline {
  inline namespace Inline {
    const int z = 10;
  }
}

namespace TestAliasName = Enclosing::Nested;
// NOTE: There is no warning on this alias.
namespace AliasWithSameName = Enclosing::Nested;

namespace TestUsingDecls {

namespace A {
void foo();
}
namespace B {
using A::foo; // <- a UsingDecl creating a UsingShadow
}

}// end namespace TestUsingDecls

namespace TestUnresolvedTypenameAndValueDecls {

template <class T> class Base;
template <class T> class Derived : public Base<T> {
public:
  using typename Base<T>::foo;
  using Base<T>::bar;
  typedef typename Derived::foo NewUnresolvedUsingType;
};

} // end namespace TestUnresolvedTypenameAndValueDecls

namespace TestUsingNamespace {
  using namespace Enclosing;
}
