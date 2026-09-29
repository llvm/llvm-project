// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order z,y,x %s -- | FileCheck %s

namespace bar {

#define INT_DECL(NAME) int NAME // CHECK:      {{^#define INT_DECL\(NAME\) int NAME}}
#define MACRO_DECL int x;       // CHECK-NEXT: {{^#define MACRO_DECL int x;}}

struct Foo {
  MACRO_DECL   // CHECK:      {{^ INT_DECL\(z\);}}
  int y;       // CHECK-NEXT: {{^ int y;}}
  INT_DECL(z); // CHECK-NEXT: {{^ MACRO_DECL}}
};

#define FOO 0 // CHECK:      {{^#define FOO 0}}
#define BAR 1 // CHECK-NEXT: {{^#define BAR 1}}
#define BAZ 2 // CHECK-NEXT: {{^#define BAZ 2}}

struct Foo foo = {
  FOO, // CHECK:      {{^ BAZ,}}
  BAR, // CHECK-NEXT: {{^ BAR,}}
  BAZ, // CHECK-NEXT: {{^ FOO,}}
};

} // end namespace bar
