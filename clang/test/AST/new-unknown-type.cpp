// RUN: %clang_cc1 -verify -ast-dump %s | FileCheck %s

extern void foo(Unknown*); // expected-error {{unknown type name 'Unknown'}}

namespace a {
  void computeSomething() {
    foo(new Unknown()); // expected-error {{unknown type name 'Unknown'}}
    foo(new Unknown{}); // expected-error {{unknown type name 'Unknown'}}
    foo(new Unknown);   // expected-error {{unknown type name 'Unknown'}}
  }
} // namespace a

namespace b {
  struct Bar{};
} // namespace b

// CHECK:      |-NamespaceDecl 0x{{[^ ]*}} <line:5:1, line:11:1> line:5:11 a
// CHECK-NEXT: | `-FunctionDecl 0x{{[^ ]*}} <line:6:3, line:10:3> line:6:8 computeSomething 'void ()'
// CHECK-NEXT: |   `-CompoundStmt 0x{{[^ ]*}} <col:27, line:10:3>
// CHECK-NEXT: |-NamespaceDecl 0x{{[^ ]*}} <line:13:1, line:15:1> line:13:11 b
// CHECK-NEXT: | `-CXXRecordDecl 0x{{[^ ]*}} <line:14:3, col:14> col:10 referenced struct Bar definition

static b::Bar bar;
// CHECK:      `-VarDecl 0x{{[^ ]*}} <line:23:1, col:15> col:15 bar 'b::Bar' static callinit
// CHECK-NEXT:   `-CXXConstructExpr 0x{{[^ ]*}} <col:15> 'b::Bar' 'void () noexcept'
