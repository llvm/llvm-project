// RUN: c-index-test -index-file -check-prefix CHECK %s -target armv7-windows-gnu -fdeclspec

void __declspec(dllexport) export_function(void) {}
// CHECK: [indexDeclaraton]: kind: function | name: export_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllexport)
void __attribute__((dllexport)) export_gnu_attribute(void) {}
// CHECK: [indexDeclaration] kind: function | name: export_gnu_attribute | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllexport)

void __declspec(dllimport) import_function(void);
// CHECK: [indexDeclaration] kind: function | name: import_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllimport)
void __attribute__((dllimport)) import_gnu_attribute(void);
// CHECK: [indexDeclaration] kind: function | name: import_gnu_function | {{.*}} | lang: C
// CHECK: <attribute>: attribute(dllimport)

