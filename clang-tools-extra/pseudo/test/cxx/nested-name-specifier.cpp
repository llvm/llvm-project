// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s

// Verify that we don't form a complete `::` nested-name-specifier if there is
// an identifier preceding it.
Foo::Foo() {} // No  "Foo ::Foo()" false parse
// CHECK:      ├─declaration-seq~function-definition := function-declarator function-body
// CHECK-NEXT: │ ├─function-declarator~noptr-declarator := noptr-declarator parameters-and-qualifiers

int ::x;
// CHECK:      declaration~simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK-NEXT: ├─decl-specifier-seq~INT

void test() {
  X::Y::Z; // No false qualified-declarator parses "X ::Y::Z" and "X::Y ::Z".
// CHECK:  statement-seq~statement := <ambiguous>
// CHECK:  statement~expression-statement := expression ;
// CHECK:  statement~simple-declaration := decl-specifier-seq ;
// CHECK-NOT: simple-declaration := decl-specifier-seq init-declarator-list ;

  // FIXME: eliminate the false `a<b> ::c` declaration parse.
  a<b>::c;
// CHECK: statement := <ambiguous>
// CHECK-NEXT: ├─statement~expression-statement := expression ;
// CHECK-NEXT: │ ├─expression~relational-expression :=
// CHECK:      └─statement~simple-declaration := <ambiguous>
// CHECK-NEXT:   ├─simple-declaration := decl-specifier-seq ;
// CHECK:        └─simple-declaration := decl-specifier-seq init-declarator-list ;
}
