// The run lines are below, because this test is line- and
// column-number sensitive.

template<typename T>
struct X {
  X(T);
};

struct Virt { };
struct Y : virtual Virt { };

struct Z : public X<int>, public Y {
  Z();

  int a, b, c;
};

Z::Z() : ::X<int>(0), Virt(), b(), c() { }

// RUN: c-index-test -code-completion-at=%s:18:10 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: MemberRef:{TypedText a}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC1: MemberRef:{TypedText b}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC1: MemberRef:{TypedText c}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC1: NotImplemented:{TypedText Virt}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC1: NotImplemented:{TypedText X<int>}{LeftParen (}{Placeholder args}{RightParen )} (7)
// CHECK-CC1: NotImplemented:{TypedText Y}{LeftParen (}{Placeholder args}{RightParen )} (35)

// RUN: c-index-test -code-completion-at=%s:18:23 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: MemberRef:{TypedText a}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC2: MemberRef:{TypedText b}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC2: MemberRef:{TypedText c}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC2: NotImplemented:{TypedText Virt}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC2: NotImplemented:{TypedText Y}{LeftParen (}{Placeholder args}{RightParen )} (7)

// RUN: c-index-test -code-completion-at=%s:18:36 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: MemberRef:{TypedText a}{LeftParen (}{Placeholder args}{RightParen )} (35)
// CHECK-CC3-NOT: MemberRef:{TypedText b}{LeftParen (}{Placeholder args}{RightParen )}
// CHECK-CC3: MemberRef:{TypedText c}{LeftParen (}{Placeholder args}{RightParen )} (7)
// CHECK-CC3-NOT: NotImplemented:{TypedText Virt}{LeftParen (}{Placeholder args}{RightParen )}
// CHECK-CC3: NotImplemented:{TypedText Y}{LeftParen (}{Placeholder args}{RightParen )} (35)
