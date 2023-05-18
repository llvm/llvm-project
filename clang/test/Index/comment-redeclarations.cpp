// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out

class Foo;
// CHECK: CXComment_Text Text=[ Foo is the best!])))]

/// Foo is the best!
class Foo;
// CHECK: CXComment_Text Text=[ Foo is the best!])))]

class Foo {};
// CHECK: CXComment_Text Text=[ Foo is the best!])))]
