// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s

// not parsed as Type{foo} Type{bar}
foo bar;
// CHECK-NOT: simple-declaration := decl-specifier-seq ;
// CHECK:    simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK:    ├─decl-specifier-seq~simple-type-specifier
// CHECK:    ├─init-declarator-list~IDENTIFIER
// CHECK:    └─;
// CHECK-NOT: simple-declaration := decl-specifier-seq ;

// not parsed as Type{std} Type{::string} Declarator{s};
std::string s;
// CHECK-NOT: nested-name-specifier := ::
// CHECK:     simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK:     ├─decl-specifier-seq~simple-type-specifier := <ambiguous>
// CHECK:     │ ├─simple-type-specifier := nested-name-specifier type-name
// CHECK:     │ │ ├─nested-name-specifier := <ambiguous> #1
// CHECK:     │ │ │ ├─nested-name-specifier := type-name ::
// CHECK:     │ │ │ └─nested-name-specifier := namespace-name ::
// CHECK:     │ │ └─type-name
// CHECK:     │ └─simple-type-specifier := nested-name-specifier template-name
// CHECK:     │   ├─nested-name-specifier =#1
// CHECK:     │   └─template-name~IDENTIFIER
// CHECK:     ├─init-declarator-list~IDENTIFIER
// CHECK:     └─;
// CHECK-NOT: nested-name-specifier := ::
