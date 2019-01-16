// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -strict-whitespace %s

struct R {
  R() = default;
  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:15> col:3 used constexpr R 'void () noexcept' default trivial
  ~R() {} // not trivial
  // CHECK: CXXDestructorDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:3 used ~R 'void () noexcept'
  R(const R&) = delete;
  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:22> col:3 R 'void (const R &)' delete trivial
  R(R&&) = default;
  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> col:3 constexpr R 'void (R &&)' default trivial noexcept-unevaluated

  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-10]]:8> col:8 implicit operator= 'R &(const R &)' inline default_delete trivial noexcept-unevaluated
};

struct S {
  int i, j;
  R r;

  S() : i(0), j(0) {}
  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:21> col:3 S 'void ()'
  // CHECK-NEXT: CXXCtorInitializer Field 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 0
  // CHECK-NEXT: CXXCtorInitializer Field 0x{{[^ ]*}} 'j' 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:17> 'int' 0
  // CHECK-NEXT: CXXCtorInitializer Field 0x{{[^ ]*}} 'r' 'R'
  // CHECK-NEXT: CXXConstructExpr 0x{{[^ ]*}} <col:3> 'R' 'void () noexcept'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:20, col:21>

  void a();
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> col:8 a 'void ()'
  void b() const;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> col:8 b 'void () const'
  void c() volatile;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> col:8 c 'void () volatile'
  void d() &;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> col:8 d 'void () &'
  void e() &&;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> col:8 e 'void () &&'
  virtual void f(float, int = 12);
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:33> col:16 f 'void (float, int)' virtual
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:18> col:23 'float'
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:25, col:31> col:29 'int' cinit
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:31> 'int' 12

  virtual void g() = 0;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:22> col:16 g 'void ()' virtual pure

  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-33]]:8> col:8 implicit S 'void (const S &)' inline default_delete noexcept-unevaluated
  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <col:8> col:8 implicit constexpr S 'void (S &&)' inline default noexcept-unevaluated
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <col:8> col:8 implicit operator= 'S &(const S &)' inline default_delete noexcept-unevaluated
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <col:8> col:8 implicit operator= 'S &(S &&)' inline default_delete noexcept-unevaluated
  // CHECK: CXXDestructorDecl 0x{{[^ ]*}} <col:8> col:8 implicit ~S 'void ()' inline default noexcept-unevaluated
};

struct T : S { // T is not referenced, but S is
  void f(float, int = 100) override;
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:28> col:8 f 'void (float, int)'
  // CHECK-NEXT: Overrides: [ 0x{{[^ ]*}} S::f 'void (float, int)' ]
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:10> col:15 'float'
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:17, col:23> col:21 'int' cinit
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:23> 'int' 100
  // CHECK-NEXT: OverrideAttr

  // CHECK: CXXConstructorDecl 0x{{[^ ]*}} <line:[[@LINE-9]]:8> col:8 implicit T 'void (const T &)' inline default_delete noexcept-unevaluated
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <col:8> col:8 implicit operator= 'T &(const T &)' inline default_delete noexcept-unevaluated
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <col:8> col:8 implicit operator= 'T &(T &&)' inline default_delete noexcept-unevaluated
  // CHECK: CXXDestructorDecl 0x{{[^ ]*}} <col:8> col:8 implicit ~T 'void ()' inline default noexcept-unevaluated
};

struct U {
  void f();
  // CHECK: CXXMethodDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> col:8 f 'void ()'
};
void U::f() {} // parent
// CHECK: CXXMethodDecl 0x{{[^ ]*}} parent 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:14> col:9 f 'void ()'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:13, col:14>

void a1();
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:9> col:6 used a1 'void ()'
void a2(void);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:13> col:6 a2 'void ()'
void b(int a, int b);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:20> col:6 b 'void (int, int)'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:8, col:12> col:12 a 'int'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:15, col:19> col:19 b 'int'
void c(int a, int b = 12);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:25> col:6 c 'void (int, int)'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:8, col:12> col:12 a 'int'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:15, col:23> col:19 b 'int' cinit
// CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:23> 'int' 12
constexpr void d(void);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:22> col:16 constexpr d 'void ()'
static void e(void);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:19> col:13 e 'void ()' static
extern void f(void);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:19> col:13 f 'void ()' extern
extern "C" void g(void);
// CHECK: LinkageSpecDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:23> col:8 C
// CHECK: FunctionDecl 0x{{[^ ]*}} <col:12, col:23> col:17 g 'void ()'
inline void h(void);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:19> col:13 h 'void ()' inline
void i(void) noexcept;
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:14> col:6 i 'void () noexcept'
void j(void) noexcept(false);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:28> col:6 j 'void () noexcept(false)'
void k(void) noexcept(1);
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:24> col:6 k 'void () noexcept(1)'
template <typename T>
T l(T&);
// CHECK: FunctionTemplateDecl 0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:7> col:3 l
// CHECK-NEXT: TemplateTypeParmDecl 0x{{[^ ]*}} <line:[[@LINE-3]]:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-3]]:1, col:7> col:3 l 'T (T &)'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:5, col:6> col:7 'T &'

void m(int) {}
// CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:14> col:6 m 'void (int)'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:8> col:11 'int'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:13, col:14>

int main() {
  // CHECK: FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:5 main 'int ()'
  a1(); // Causes this to be marked 'used'
}
