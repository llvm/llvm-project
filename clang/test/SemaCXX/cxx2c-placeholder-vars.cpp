// RUN: %clang_cc1 -fsyntax-only -verify -ast-dump -std=c++2c -Wunused-parameter -Wunused -Wpre-c++26-compat %s | FileCheck %s

void static_var() {
    static int _; // expected-note {{previous definition is here}} \
                  // expected-note {{candidate}}
    static int _; // expected-error {{redefinition of '_'}}
    int _;        // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
                  // expected-note {{candidate}}
    _++; // expected-error{{reference to '_' is ambiguous}}
}

void static_var_2() {
    int _; // expected-note {{previous definition is here}}
    static int _; // expected-error {{redefinition of '_'}}
}

void bindings() {
    int arr[4] = {0, 1, 2, 3};
    auto [_, _, _, _] = arr; // expected-warning 3{{placeholder variables are incompatible with C++ standards before C++2c}} \
                             // expected-note 4{{placeholder declared here}}
    _ == 42; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    {
        // no extension warning as we only introduce a single placeholder.
        auto [_, a, b, c] = arr; // expected-warning {{unused variable '[_, a, b, c]'}}
    }
    {
        auto [_, _, b, c] = arr; // expected-warning {{unused variable '[_, _, b, c]'}} \
                                 // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}}
    }
    {
        // There are only 3 extension warnings because the first
        // introduction of `_` is valid in all C++ standards
        auto [_, _, _, _] = arr; // expected-warning 3{{placeholder variables are incompatible with C++ standards before C++2c}}
    }
}

namespace StaticBindings {

int arr[2] = {0, 1};
static auto [_, _] = arr; // expected-error {{redefinition of '_'}} \
                          // expected-note  {{previous definition is here}}

void f() {
    int arr[2] = {0, 1};
    static auto [_, _] = arr; // expected-error {{redefinition of '_'}} \
                            // expected-note  {{previous definition is here}}
}

}

void lambda() {
    (void)[_ = 0, _ = 1] { // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
                           // expected-note 2{{placeholder declared here}}
        (void)_++; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    };

    {
        int _ = 12;
        (void)[_ = 0]{ return _;}; // no warning (different scope)
    }

    auto GH107024 = [_ = 42]() { return _; }();
}

namespace global_var {
    int _; // expected-note {{previous definition is here}}
    int _; // expected-error {{redefinition of '_'}}
}

namespace {
    int _; // expected-note {{previous definition is here}}
    int _; // expected-error {{redefinition of '_'}}
}


namespace global_fun {
void _();
void _();

void _() {} // expected-note {{previous definition is here}}
void _() {} // expected-error {{redefinition of '_'}}
void _(int){}
}

typedef int _;
typedef int _; // Type redeclaration, nothing to do with placeholders

void extern_test() {
    extern int _;
    extern int _; // expected-note {{candidate}}
    int _; //expected-note {{candidate}}
    _++; // expected-error {{reference to '_' is ambiguous}}
}


struct Members {
    int _; // expected-note 2{{placeholder declared here}}
    int _; // expected-warning{{placeholder variables are incompatible with C++ standards before C++2c}} \
           // expected-note 2{{placeholder declared here}}
    void f() {
        _++; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    }
    void attributes() __attribute__((diagnose_if(_ != 0, "oh no!", "warning"))); // expected-error{{ambiguous reference to placeholder '_', which is defined multiple times}}
};

namespace using_ {
int _; // expected-note {{target of using declaration}}
void f() {
    int _; // expected-note {{conflicting declaration}}
    _ = 0;
    using using_::_; // expected-error {{target of using declaration conflicts with declaration already in scope}}
}
}


void call(int);
void test_param(int _) {}
void test_params(int _, int _); // expected-error {{redefinition of parameter '_'}} \
                                // expected-note {{previous declaration is here}}

template <auto _, auto _> // expected-error {{declaration of '_' shadows template parameter}} \
                          // expected-note  {{template parameter is declared here}}
auto i = 0;

template <typename T>
concept C = requires(T _, T _) {  // expected-error {{redefinition of parameter '_'}} \
                                // expected-note {{previous declaration is here}}
    T{};
};

struct S {
    int a;
};

void f(S a, S _) { // expected-warning {{unused parameter 'a'}}

}

void unused_warning() {
  int _ = 12; // placeholder variable, no unused-but-set warning
  int x = 12; // expected-warning {{unused variable 'x'}}
  int _ = 12; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}}
}

struct ShadowMembers {
  int _;
  void f() {
    int _;
    _ = 12; // Ok, access the local variable
    (void)({ int _ = 12; _;}); // Ok, inside a different scope
  }
};

struct MemberPtrs {
  int _, _; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
            // expected-note 4{{placeholder declared here}}
};
constexpr int oh_no = __builtin_offsetof(MemberPtrs, _); // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
int MemberPtrs::* ref = &MemberPtrs::_; // expected-error{{ambiguous reference to placeholder '_', which is defined multiple times}}


struct MemberInitializer {
  MemberInitializer() : _(0) {}  // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
  int _, _; // expected-note 2{{placeholder declared here}} \
            // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}}
};

struct MemberAndUnion {
  int _; // expected-note {{placeholder declared here}}
  union { int _; int _; }; // expected-note 2 {{placeholder declared here}} \
                           // expected-warning 2{{placeholder variables are incompatible with C++ standards before C++2c}}


  MemberAndUnion() : _(0) {} // expected-error {{ambiguous reference to placeholder '_', which is defined multiple time}}
};

struct Union { union { int _, _, _; }; };   // expected-note 3{{placeholder declared here}} \
                                            // expected-warning 2{{placeholder variables are incompatible with C++ standards before C++2c}}

void TestUnion() {
   Union c;
   c._ = 0; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
}

void AnonymousLocals() {
    union  {int _, _;}; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}}  \
                        // expected-note 2{{placeholder declared here}}
    union  {int _, _;}; // expected-warning 2{{placeholder variables are incompatible with C++ standards before C++2c}} \
                        // expected-note 2{{placeholder declared here}}
    _. = 0; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
}

namespace StaticUnions {

static union { int _ = 42; }; // expected-note {{previous declaration is here}}
static union { int _ = 43; }; // expected-error {{member of anonymous union redeclares '_'}}

inline void StaticUnion() {
  static union { int _{}; };  // expected-note {{previous declaration is here}}
  static union { int _{}; }; // expected-error {{member of anonymous union redeclares '_'}}
}

}

namespace TagVariables {

[[maybe_unused]] struct {
    int _, _, _;  // expected-warning 2{{placeholder variables are incompatible with C++ standards before C++2c}}
} a;

[[maybe_unused]] union {
    int _, _, _; // expected-warning 2{{placeholder variables are incompatible with C++ standards before C++2c}}
} b;

}

namespace MemberLookupTests {

struct S {
    int _, _; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
              // expected-note 8{{placeholder declared here}}

    void f() {
        _ ++ ; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    }
};

struct T : S {

};

void Test() {
    S s{._ =0}; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    S{}._; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
    T{}._; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
};

};

namespace Bases {
    struct S {
        int _, _; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
                  // expected-note   2{{placeholder declared here}}
        int a;
    };
    struct T : S{
        int _, _; // expected-warning {{placeholder variables are incompatible with C++ standards before C++2c}} \
                  // expected-note   2{{placeholder declared here}}
        int a;
        void f() {
            _; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}}
            S::_; // expected-error {{ambiguous reference to placeholder '_', which is defined multiple times}} \
                  // expected-error {{a type specifier is required for all declarations}}
        }
    };
}

namespace GH114069 {

template <class T>
struct A {
    T _ = 1;
    T _ = 2;
    T : 1;
    T a = 3;
    T _ = 4;
};

void f() {
    [[maybe_unused]] A<int> a;
}

// CHECK: NamespaceDecl {{.*}} GH114069
// CHECK: ClassTemplateSpecializationDecl {{.*}} struct A definition
// CHECK: CXXConstructorDecl {{.*}} implicit used constexpr A 'void () noexcept'
// CHECK-NEXT: CXXCtorInitializer Field {{.*}} '_' 'int'
// CHECK-NEXT: CXXDefaultInitExpr {{.*}} 'int' has rewritten init
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: CXXCtorInitializer Field {{.*}} '_' 'int'
// CHECK-NEXT: CXXDefaultInitExpr {{.*}} 'int' has rewritten init
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: CXXCtorInitializer Field {{.*}} 'a' 'int'
// CHECK-NEXT: CXXDefaultInitExpr {{.*}} 'int' has rewritten init
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: CXXCtorInitializer Field {{.*}} '_' 'int'
// CHECK-NEXT: CXXDefaultInitExpr {{.*}} 'int' has rewritten init
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: CompoundStmt {{.*}}

}
