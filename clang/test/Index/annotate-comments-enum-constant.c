// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s --dump-input always < %t/out

enum {
  /// Documentation for Foo
  Foo,
  Bar, // No documentation for Bar
  /// Documentation for Baz
  Baz,
};
// CHECK: EnumConstantDecl=Foo:[[@LINE-5]]:3 (Definition) {{.*}} BriefComment=[Documentation for Foo] FullCommentAsHTML=[<p class="para-brief"> Documentation for Foo</p>] FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments-enum-constant.c" line="[[@LINE-5]]" column="3"><Name>Foo</Name><USR>c:@Ea@Foo@Foo</USR><Declaration>Foo</Declaration><Abstract><Para> Documentation for Foo</Para></Abstract></Variable>]
// CHECK: EnumConstantDecl=Bar:[[@LINE-5]]:3 (Definition)
// CHECK-NOT: BriefComment=[Documentation for Foo]
// CHECK: EnumConstantDecl=Baz:[[@LINE-5]]:3 (Definition) {{.*}} BriefComment=[Documentation for Baz] FullCommentAsHTML=[<p class="para-brief"> Documentation for Baz</p>] FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments-enum-constant.c" line="[[@LINE-5]]" column="3"><Name>Baz</Name><USR>c:@Ea@Foo@Baz</USR><Declaration>Baz</Declaration><Abstract><Para> Documentation for Baz</Para></Abstract></Variable>]

