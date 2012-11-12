class Base {
public:
  virtual void f();
};

class Derived final : public Base {
  virtual void f() override final;

  struct final { };
};

typedef int override;
struct Base2 {
  virtual int override();
};

struct Derived2 : Base2 {
  ::override override() override;
};

// RUN: c-index-test -test-annotate-tokens=%s:6:1:19:1 %s | FileCheck -check-prefix=CHECK-OVERRIDE-FINAL %s

// CHECK-OVERRIDE-FINAL: Keyword: "class" [6:1 - 6:6] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Identifier: "Derived" [6:7 - 6:14] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Keyword: "final" [6:15 - 6:20] attribute(final)=
// CHECK-OVERRIDE-FINAL: Punctuation: ":" [6:21 - 6:22] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Keyword: "public" [6:23 - 6:29] C++ base class specifier=class Base:1:7 [access=public isVirtual=false]
// CHECK-OVERRIDE-FINAL: Identifier: "Base" [6:30 - 6:34] TypeRef=class Base:1:7
// CHECK-OVERRIDE-FINAL: Punctuation: "{" [6:35 - 6:36] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Keyword: "virtual" [7:3 - 7:10] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Keyword: "void" [7:11 - 7:15] CXXMethod=f:7:16 (virtual) [Overrides @3:16]
// CHECK-OVERRIDE-FINAL: Identifier: "f" [7:16 - 7:17] CXXMethod=f:7:16 (virtual) [Overrides @3:16]
// CHECK-OVERRIDE-FINAL: Punctuation: "(" [7:17 - 7:18] CXXMethod=f:7:16 (virtual) [Overrides @3:16]
// CHECK-OVERRIDE-FINAL: Punctuation: ")" [7:18 - 7:19] CXXMethod=f:7:16 (virtual) [Overrides @3:16]
// CHECK-OVERRIDE-FINAL: Keyword: "override" [7:20 - 7:28] attribute(override)=
// CHECK-OVERRIDE-FINAL: Keyword: "final" [7:29 - 7:34] attribute(final)=
// CHECK-OVERRIDE-FINAL: Punctuation: ";" [7:34 - 7:35] ClassDecl=Derived:6:7 (Definition)
// CHECK-OVERRIDE-FINAL: Keyword: "struct" [9:3 - 9:9] StructDecl=final:9:10 (Definition)
// CHECK-OVERRIDE-FINAL: Identifier: "final" [9:10 - 9:15] StructDecl=final:9:10 (Definition)
// CHECK-OVERRIDE-FINAL: Punctuation: "{" [9:16 - 9:17] StructDecl=final:9:10 (Definition)
// CHECK-OVERRIDE-FINAL: Punctuation: "}" [9:18 - 9:19] StructDecl=final:9:10 (Definition)
// CHECK-OVERRIDE-FINAL: Punctuation: ";" [9:19 - 9:20] ClassDecl=Derived:6:7 (Definition)
