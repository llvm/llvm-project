// Code-completion through the C interface
#include "nonexistent_header.h"
struct X {
  int member;
  
  enum E { Val1 };
};

struct Y {
  float member;
  void memfunc(int i = 17);
};

struct Z : X, Y {
  double member;
  operator int() const;
};
struct W { };
struct Z get_Z();
namespace N { }
void test_Z() {
  // RUN: c-index-test -code-completion-at=%s:23:11 %s | FileCheck -check-prefix=CHECK-MEMBER %s
  get_Z().member = 17;
}


float& overloaded(int i, long second);
double& overloaded(float f, int second);
int& overloaded(Z z, int second);
                
void test_overloaded() {
  // RUN: c-index-test -code-completion-at=%s:33:18 %s | FileCheck -check-prefix=CHECK-OVERLOAD %s
  overloaded(Z(), 0);
}

Z::operator int() const {
  return 0;
}

// CHECK-MEMBER: FieldDecl:{ResultType double}{TypedText member}
// CHECK-MEMBER: FieldDecl:{ResultType int}{Text X::}{TypedText member}
// CHECK-MEMBER: FieldDecl:{ResultType float}{Text Y::}{TypedText member}
// CHECK-MEMBER: CXXMethod:{ResultType void}{Informative Y::}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )}
// CHECK-MEMBER: CXXConversion:{TypedText operator int}{LeftParen (}{RightParen )}{Informative  const}
// CHECK-MEMBER: CXXMethod:{ResultType Z &}{TypedText operator=}{LeftParen (}{Placeholder const Z &}{RightParen )}
// CHECK-MEMBER: CXXMethod:{ResultType X &}{Text X::}{TypedText operator=}{LeftParen (}{Placeholder const X &}{RightParen )}
// CHECK-MEMBER: CXXMethod:{ResultType Y &}{Text Y::}{TypedText operator=}{LeftParen (}{Placeholder const Y &}{RightParen )}
// CHECK-MEMBER: EnumConstantDecl:{ResultType X::E}{Informative E::}{TypedText Val1}
// CHECK-MEMBER: StructDecl:{TypedText X}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Y}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Z}{Text ::}
// CHECK-MEMBER: CXXDestructor:{ResultType void}{Informative X::}{TypedText ~X}{LeftParen (}{RightParen )}
// CHECK-MEMBER: CXXDestructor:{ResultType void}{Informative Y::}{TypedText ~Y}{LeftParen (}{RightParen )}
// CHECK-MEMBER: CXXDestructor:{ResultType void}{TypedText ~Z}{LeftParen (}{RightParen )}
// CHECK-MEMBER: Completion contexts:
// CHECK-MEMBER-NEXT: Dot member access
// CHECK-MEMBER-NEXT: Container Kind: StructDecl
// CHECK-MEMBER-NEXT: Container is complete
// CHECK-MEMBER-NEXT: Container USR: c:@S@Z

// CHECK-OVERLOAD: OverloadCandidate:{ResultType int &}{Text overloaded}{LeftParen (}{Placeholder Z z}{Comma , }{CurrentParameter int second}{RightParen )}
// CHECK-OVERLOAD: OverloadCandidate:{ResultType float &}{Text overloaded}{LeftParen (}{Placeholder int i}{Comma , }{CurrentParameter long second}{RightParen )}
// CHECK-OVERLOAD: OverloadCandidate:{ResultType double &}{Text overloaded}{LeftParen (}{Placeholder float f}{Comma , }{CurrentParameter int second}{RightParen )}
// CHECK-OVERLOAD: Completion contexts:
// CHECK-OVERLOAD-NEXT: Any type
// CHECK-OVERLOAD-NEXT: Any value
// CHECK-OVERLOAD-NEXT: Enum tag
// CHECK-OVERLOAD-NEXT: Union tag
// CHECK-OVERLOAD-NEXT: Struct tag
// CHECK-OVERLOAD-NEXT: Class name
// CHECK-OVERLOAD-NEXT: Nested name specifier
// CHECK-OVERLOAD-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:37:10 %s | FileCheck -check-prefix=CHECK-EXPR %s
// CHECK-EXPR: NotImplemented:{TypedText int} (50)
// CHECK-EXPR: NotImplemented:{TypedText long} (50)
// CHECK-EXPR: FieldDecl:{ResultType double}{TypedText member} (17)
// CHECK-EXPR: FieldDecl:{ResultType int}{Text X::}{TypedText member} (9)
// CHECK-EXPR: FieldDecl:{ResultType float}{Text Y::}{TypedText member} (18)
// CHECK-EXPR: CXXMethod:{ResultType void}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )} (37)
// CHECK-EXPR: Namespace:{TypedText N}{Text ::} (75)
// CHECK-EXPR: Completion contexts:
// CHECK-EXPR-NEXT: Any type
// CHECK-EXPR-NEXT: Any value
// CHECK-EXPR-NEXT: Enum tag
// CHECK-EXPR-NEXT: Union tag
// CHECK-EXPR-NEXT: Struct tag
// CHECK-EXPR-NEXT: Class name
// CHECK-EXPR-NEXT: Nested name specifier
// CHECK-EXPR-NEXT: Objective-C interface
