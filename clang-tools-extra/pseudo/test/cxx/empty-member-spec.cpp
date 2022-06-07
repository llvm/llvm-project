// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
class Foo {
public:
};
// CHECK:      decl-specifier-seq~class-specifier := class-head { member-specification }
// CHECK-NEXT: ├─class-head := class-key class-head-name
// CHECK-NEXT: │ ├─class-key~CLASS := tok[0]
// CHECK-NEXT: │ └─class-head-name~IDENTIFIER := tok[1]
// CHECK-NEXT: ├─{ := tok[2]
// CHECK-NEXT: ├─member-specification := access-specifier :
// CHECK-NEXT: │ ├─access-specifier~PUBLIC := tok[3]
// CHECK-NEXT: │ └─: := tok[4]
// CHECK-NEXT: └─} := tok[5]
