// RUN: c-index-test -index-file -check-prefix CHECK %s -target armv7-windows-gnu -fdeclspec

struct __declspec(dllexport) export_s {
  void m();
};
// CHECK: [indexDeclaration]: kind: struct | name: export_s | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllexport)
// CHECK: [indexDeclaration]: kind: c++-instance-method | name: m | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllexport)

struct __declspec(dllimport) import_s {
  void m();
};
// CHECK: [indexDeclaration]: kind: struct | name: import_s | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllimport)
// CHECK: [indexDeclaration]: kind: c++-instance-method | name: m | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllimport)

class __attribute__((dllexport)) export_gnu_s {
  void m();
};
// CHECK: [indexDeclaration]: kind: struct | name: export_gnu_s | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllexport)
// CHECK: [indexDeclaration]: kind: c++-instance-method | name: m | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllexport)

class __attribute__((dllimport)) import_gnu_s {
  void m();
};
// CHECK: [indexDeclaration]: kind: struct | name: import_gnu_s | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllimport)
// CHECK: [indexDeclaration]: kind: c++-instance-method | name: m | {{.*}} | lang: C++
// CHECK: <attribute>: attribute(dllimport)

extern "C" void __declspec(dllexport) export_function(void) {}
// CHECK: [indexDeclaraton]: kind: function | name: export_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllexport)
extern "C" void __attribute__((dllexport)) export_gnu_function(void) {}
// CHECK: [indexDeclaraton]: kind: function | name: export_gnu_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllexport)

extern "C" {
void __declspec(dllimport) import_function(void);
// CHECK: [indexDeclaration] kind: function | name: import_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllimport)
void __attribute__((dllimport)) import_gnu_function(void);
// CHECK: [indexDeclaration] kind: function | name: import_gnu_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllimport)
}

