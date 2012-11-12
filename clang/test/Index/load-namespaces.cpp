// Test is line- and column-sensitive; see below.

namespace std {
  namespace rel_ops {
    void f();
  }
}

namespace std {
  void g();
}

namespace std98 = std;
namespace std0x = std98;

using namespace std0x;

namespace std {
  int g(int);
}

using std::g;

void std::g() {
}

namespace my_rel_ops = std::rel_ops;

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-namespaces.cpp:3:11: Namespace=std:3:11 (Definition) Extent=[3:1 - 7:2]
// CHECK: load-namespaces.cpp:4:13: Namespace=rel_ops:4:13 (Definition) Extent=[4:3 - 6:4]
// CHECK: load-namespaces.cpp:5:10: FunctionDecl=f:5:10 Extent=[5:5 - 5:13]
// CHECK: load-namespaces.cpp:9:11: Namespace=std:9:11 (Definition) Extent=[9:1 - 11:2]
// CHECK: load-namespaces.cpp:10:8: FunctionDecl=g:10:8 Extent=[10:3 - 10:11]
// CHECK: load-namespaces.cpp:13:11: NamespaceAlias=std98:13:11 Extent=[13:1 - 13:22]
// CHECK: load-namespaces.cpp:13:19: NamespaceRef=std:3:11 Extent=[13:19 - 13:22]
// CHECK: load-namespaces.cpp:14:11: NamespaceAlias=std0x:14:11 Extent=[14:1 - 14:24]
// CHECK: load-namespaces.cpp:14:19: NamespaceRef=std98:13:11 Extent=[14:19 - 14:24]
// CHECK: load-namespaces.cpp:16:17: UsingDirective=:16:17 Extent=[16:1 - 16:22]
// CHECK: load-namespaces.cpp:16:17: NamespaceRef=std0x:14:11 Extent=[16:17 - 16:22]
// CHECK: load-namespaces.cpp:18:11: Namespace=std:18:11 (Definition) Extent=[18:1 - 20:2]
// CHECK: load-namespaces.cpp:19:7: FunctionDecl=g:19:7 Extent=[19:3 - 19:13]
// CHECK: load-namespaces.cpp:19:12: ParmDecl=:19:12 (Definition) Extent=[19:9 - 19:13]
// CHECK: load-namespaces.cpp:22:12: UsingDeclaration=g[19:7, 10:8] Extent=[22:1 - 22:13]
// CHECK: load-namespaces.cpp:22:7: NamespaceRef=std:18:11 Extent=[22:7 - 22:10]
// CHECK: load-namespaces.cpp:24:11: FunctionDecl=g:24:11 (Definition) Extent=[24:1 - 25:2]
// CHECK: load-namespaces.cpp:24:6: NamespaceRef=std:18:11 Extent=[24:6 - 24:9]
// CHECK: load-namespaces.cpp:27:11: NamespaceAlias=my_rel_ops:27:11 Extent=[27:1 - 27:36]
// CHECK: load-namespaces.cpp:27:24: NamespaceRef=std:18:11 Extent=[27:24 - 27:27]
// CHECK: load-namespaces.cpp:27:29: NamespaceRef=rel_ops:4:13 Extent=[27:29 - 27:36]
