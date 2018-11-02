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

struct PR23948 {
  template<class size> PR23948()
        :
  {}

  template<class size> void invalid()
        :
  {}

  int a;
};

// RUN: c-index-test -code-completion-at=%s:18:10 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: MemberRef:{TypedText a}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC1: MemberRef:{TypedText b}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC1: MemberRef:{TypedText c}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC1: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder const Virt &}{RightParen )} (35)
// CHECK-CC1: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder Virt &&}{RightParen )} (35)
// CHECK-CC1: CXXConstructor:{TypedText X<int>}{LeftParen (}{Placeholder int}{RightParen )} (7)
// CHECK-CC1: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder const Y &}{RightParen )} (35)
// CHECK-CC1: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder Y &&}{RightParen )} (35)

// RUN: c-index-test -code-completion-at=%s:18:23 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: MemberRef:{TypedText a}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC2: MemberRef:{TypedText b}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC2: MemberRef:{TypedText c}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC2: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder const Virt &}{RightParen )} (35)
// CHECK-CC2: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder Virt &&}{RightParen )} (35)
// CHECK-CC2: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder const Y &}{RightParen )} (7)
// CHECK-CC2: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder Y &&}{RightParen )} (7)

// RUN: c-index-test -code-completion-at=%s:18:36 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: MemberRef:{TypedText a}{LeftParen (}{Placeholder int}{RightParen )} (35)
// CHECK-CC3-NOT: MemberRef:{TypedText b}{LeftParen (}{Placeholder int}{RightParen )}
// CHECK-CC3: MemberRef:{TypedText c}{LeftParen (}{Placeholder int}{RightParen )} (7)
// CHECK-CC3-NOT: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder const Virt &}{RightParen )} (35)
// CHECK-CC3-NOT: CXXConstructor:{TypedText Virt}{LeftParen (}{Placeholder Virt &&}{RightParen )} (35)
// CHECK-CC3: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder const Y &}{RightParen )} (35)
// CHECK-CC3: CXXConstructor:{TypedText Y}{LeftParen (}{Placeholder Y &&}{RightParen )} (35)

// RUN: c-index-test -code-completion-at=%s:22:10 -target i386-apple-darwin %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: MemberRef:{TypedText a}{LeftParen (}{Placeholder int}{RightParen )} (7)
// RUN: c-index-test -code-completion-at=%s:26:10 %s
