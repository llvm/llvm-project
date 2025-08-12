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

template <typename Key, typename Value>
struct map {
  Value& operator[](Key&& p);
  Value& operator[](const Key& p);
};
} // namespace std

// CHECK-NOT:   LifetimeCaptureByAttr

struct [[gsl::Pointer()]] View {};
std::vector<View> views;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const View &)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'const View &'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (View &&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'View &&'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, View &&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'iterator'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'View &&'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit

template <class T> struct [[gsl::Pointer()]] ViewTemplate {};
std::vector<ViewTemplate<int>> templated_views;
// CHECK:       ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const ViewTemplate<int> &)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'const ViewTemplate<int> &'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (ViewTemplate<int> &&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'ViewTemplate<int> &&'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, ViewTemplate<int> &&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'iterator'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'ViewTemplate<int> &&'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit

std::vector<int*> pointers;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int *const &)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'int *const &'
// CHECK-NOT:               LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int *&&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'int *&&'
// CHECK-NOT:               LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, int *&&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'iterator'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'int *&&'
// CHECK-NOT:               LifetimeCaptureByAttr

std::vector<int> ints;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct vector definition implicit_instantiation
// CHECK:       TemplateArgument type 'int'

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (const int &)'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} push_back 'void (int &&)'
// CHECK-NOT:   LifetimeCaptureByAttr

// CHECK:       CXXMethodDecl {{.*}} insert 'void (iterator, int &&)'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'iterator'
// CHECK-NEXT:           ParmVarDecl {{.*}} 'int &&'
// CHECK-NOT:   LifetimeCaptureByAttr

std::map<View, int> map;
// CHECK:   ClassTemplateSpecializationDecl {{.*}} struct map definition implicit_instantiation

// CHECK:       CXXMethodDecl {{.*}} operator[] 'int &(View &&)' implicit_instantiation
// CHECK-NEXT:           ParmVarDecl {{.*}} p 'View &&'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit
// CHECK:       CXXMethodDecl {{.*}} operator[] 'int &(const View &)' implicit_instantiation
// CHECK-NEXT:           ParmVarDecl {{.*}} p 'const View &'
// CHECK-NEXT:               LifetimeCaptureByAttr {{.*}} Implicit
