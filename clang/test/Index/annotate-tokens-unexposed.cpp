// RUN: c-index-test -test-annotate-tokens=%s:1:1:16:1 %s -target x86_64-pc-windows-msvc | FileCheck %s
class Foo
{
public:
    void step(int v);
    Foo();
};

void bar()
{
    // Introduce a MSInheritanceAttr node on the CXXRecordDecl for Foo. The
    // existance of this attribute should not mark all cursors for tokens in
    // Foo as UnexposedAttr.
    &Foo::step;
}

Foo::Foo()
{}

// CHECK-NOT: UnexposedAttr=
