template <typename X, typename Y>
class C
{
};

class Foo
{
public:
  C<Foo, class Bar> c;
};

void foo()
{
  Foo::

// RUN: c-index-test -code-completion-at=%s:14:8 %s -o - | FileCheck -check-prefix=CC1 %s
// CHECK-CC1: FieldDecl:{ResultType C<Foo, class Bar>}{TypedText c} (35)
// CHECK-CC1: ClassDecl:{TypedText Foo} (35)
// CHECK-CC1: CXXMethod:{ResultType Foo &}{TypedText operator=}{LeftParen (}{Placeholder const Foo &}{RightParen )}
// CHECK-CC1: CXXDestructor:{ResultType void}{TypedText ~Foo}{LeftParen (}{RightParen )} (35)
