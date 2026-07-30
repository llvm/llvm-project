namespace std {
  template<typename T>
  class allocator { 
  public:
    void in_base();
  };
  
  template<typename T, typename Alloc = std::allocator<T> >
  class vector : Alloc {
  public:
    void foo();
    void stop();
  };
  template<typename Alloc> class vector<bool, Alloc>;
}

void f() {
  std::vector<int> v;
  v.foo();
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: allocator<<#typename T#>>
  // CHECK-CC1-NEXT: vector<<#typename T#>{#, <#typename Alloc#>#}>
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-4):5 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: foo
  // CHECK-CC2: in_base
  // CHECK-CC2: stop
}


template <typename> struct X;
template <typename T> struct X<T*> { X(double); };
X<int*> x(42);
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):11 %s -o - | FileCheck -check-prefix=CHECK-CONSTRUCTOR %s
// CHECK-CONSTRUCTOR: OVERLOAD: X(<#double#>)
// (rather than X<type-parameter-0-0 *>(<#double#>)
