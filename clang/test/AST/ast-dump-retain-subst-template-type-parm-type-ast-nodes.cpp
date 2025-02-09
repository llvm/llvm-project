// RUN: %clang_cc1 -fsyntax-only -fretain-subst-template-type-parm-type-ast-nodes -ast-dump -ast-dump-filter=dump %s | FileCheck -strict-whitespace %s

namespace t1 {
template<class T> using X = T;
using dump = X<int>;

// CHECK-LABEL: Dumping t1::dump:
// CHECK-NEXT:  TypeAliasDecl
// CHECK-NEXT:  `-ElaboratedType
// CHECK-NEXT:    `-TemplateSpecializationType
// CHECK-NEXT:      |-name: 'X':'t1::X' qualified
// CHECK-NEXT:      | `-TypeAliasTemplateDecl
// CHECK-NEXT:      |-TemplateArgument
// CHECK-NEXT:      | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:      `-SubstTemplateTypeParmType 0x{{[0-9a-f]+}} 'int' sugar class depth 0 index 0 T
// CHECK-NEXT:        |-TypeAliasTemplate {{.+}} 'X'
// CHECK-NEXT:        `-BuiltinType {{.+}} 'int'
} // namespace t1
