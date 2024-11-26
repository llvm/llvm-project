// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

// Verify that we print the [[clang::lifetime_capture_by(X)]] attribute.

struct S {
    void foo(int &a, int &b) [[clang::lifetime_capture_by(a, b, global)]];
};

// CHECK: CXXMethodDecl {{.*}}clang::lifetime_capture_by(a, b, global)

// ****************************************************************************
// Infer annotation for STL container methods.
// ****************************************************************************
namespace __gnu_cxx {
template <typename T>
struct basic_iterator {};
}

namespace std {
template<typename T> class allocator {};
template <typename T, typename Alloc = allocator<T>>
struct vector {
  typedef __gnu_cxx::basic_iterator<T> iterator;
  iterator begin();

  vector();

  void push_back(const T&);
  void push_back(T&&);

  void insert(iterator, T&&);
};
} // namespace std

// CHECK-NOT:   LifetimeCaptureByAttr

struct [[gsl::Pointer()]] View {};
std::vector<View> views;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation
// CHECK:       TemplateArgument type 'View'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const View &)'
// CHECK:           ParmVarDecl {{.*}} 'const View &'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (View &&)'
// CHECK:           ParmVarDecl {{.*}} 'View &&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, View &&)'
// CHECK:           ParmVarDecl {{.*}} 'iterator'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK:           ParmVarDecl {{.*}} 'View &&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

template <class T> struct [[gsl::Pointer()]] ViewTemplate {};
std::vector<ViewTemplate<int>> templated_views;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation
// CHECK:       TemplateArgument type 'ViewTemplate<int>'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const ViewTemplate<int> &)'
// CHECK:           ParmVarDecl {{.*}} 'const ViewTemplate<int> &'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (ViewTemplate<int> &&)'
// CHECK:           ParmVarDecl {{.*}} 'ViewTemplate<int> &&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, ViewTemplate<int> &&)'
// CHECK:           ParmVarDecl {{.*}} 'iterator'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK:           ParmVarDecl {{.*}} 'ViewTemplate<int> &&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

std::vector<int*> pointers;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation
// CHECK:       TemplateArgument type 'int *'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int *const &)'
// CHECK:           ParmVarDecl {{.*}} 'int *const &'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int *&&)'
// CHECK:           ParmVarDecl {{.*}} 'int *&&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, int *&&)'
// CHECK:           ParmVarDecl {{.*}} 'iterator'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK:           ParmVarDecl {{.*}} 'int *&&'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

std::vector<int> ints;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation
// CHECK:       TemplateArgument type 'int'

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const int &)'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int &&)'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, int &&)'
// CHECK:           ParmVarDecl {{.*}} 'iterator'
// CHECK:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr
