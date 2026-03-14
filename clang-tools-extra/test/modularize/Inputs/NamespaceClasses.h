// Define same class name in different namespaces.

namespace Namespace1 {
  class NamespaceClass {
  public:
    NamespaceClass() : Member(0) {}
  private:
    int Member;
  };
}

namespace Namespace2 {
  class NamespaceClass {
  public:
    NamespaceClass() : Member(0) {}
  private:
    int Member;
  };
}

