// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -std=c++11 -Wmicrosoft -verify -fms-compatibility -fexceptions -fcxx-exceptions -fms-compatibility-version=19.00
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -std=c++11 -Wmicrosoft -verify -fms-compatibility -fexceptions -fcxx-exceptions -fms-compatibility-version=18.00

#if defined(_HAS_CHAR16_T_LANGUAGE_SUPPORT) && _HAS_CHAR16_T_LANGUAGE_SUPPORT
char16_t x;
char32_t y;
#else
typedef unsigned short char16_t;
typedef unsigned int char32_t;
#endif

_Atomic(int) z;
template <typename T>
struct _Atomic {
  _Atomic() {}
  ~_Atomic() {}
};
template <typename T>
struct atomic : _Atomic<T> {
  typedef _Atomic<T> TheBase;
  TheBase field;
};
_Atomic(int) alpha;

typename decltype(3) a; // expected-warning {{expected a qualified name after 'typename'}}

namespace ms_conversion_rules {

void f(float a);
void f(int a);

void test()
{
    long a = 0;
    f((long)0);
	f(a);
}

}


namespace ms_predefined_types {
  // ::type_info is a built-in forward class declaration.
  void f(const type_info &a);
  void f(size_t);
}


namespace ms_protected_scope {
  struct C { C(); };

  int jump_over_variable_init(bool b) {
    if (b)
      goto foo; // expected-warning {{jump from this goto statement to its label is a Microsoft extension}}
    C c; // expected-note {{jump bypasses variable initialization}}
  foo:
    return 1;
  }

struct Y {
  ~Y();
};

void jump_over_var_with_dtor() {
  goto end; // expected-warning{{jump from this goto statement to its label is a Microsoft extension}}
  Y y; // expected-note {{jump bypasses variable with a non-trivial destructor}}
 end:
    ;
}

  void jump_over_variable_case(int c) {
    switch (c) {
    case 0:
      int x = 56; // expected-note {{jump bypasses variable initialization}}
    case 1:       // expected-error {{cannot jump}}
      x = 10;
    }
  }

 
void exception_jump() {
  goto l2; // expected-error {{cannot jump}}
  try { // expected-note {{jump bypasses initialization of try block}}
     l2: ;
  } catch(int) {
  }
}

int jump_over_indirect_goto() {
  static void *ps[] = { &&a0 };
  goto *&&a0; // expected-warning {{jump from this goto statement to its label is a Microsoft extension}}
  int a = 3; // expected-note {{jump bypasses variable initialization}}
 a0:
  return 0;
}
  
}

namespace PR11826 {
  struct pair {
    pair(int v) { }
#if _MSC_VER >= 1900
    void operator=(pair&& rhs) { } // expected-note {{copy constructor is implicitly deleted because 'pair' has a user-declared move assignment operator}}
#else
    void operator=(pair&& rhs) { }
#endif
  };
  void f() {
    pair p0(3);
#if _MSC_VER >= 1900
    pair p = p0; // expected-error {{call to implicitly-deleted copy constructor of 'PR11826::pair'}}
#else
    pair p = p0;
#endif
  }
}

namespace PR11826_for_symmetry {
  struct pair {
    pair(int v) { }
#if _MSC_VER >= 1900
    pair(pair&& rhs) { } // expected-note {{copy assignment operator is implicitly deleted because 'pair' has a user-declared move constructor}}
#else
    pair(pair&& rhs) { }
#endif
  };
  void f() {
    pair p0(3);
    pair p(4);
#if _MSC_VER >= 1900
    p = p0; // expected-error {{object of type 'PR11826_for_symmetry::pair' cannot be assigned because its copy assignment operator is implicitly deleted}}
#else
    p = p0;
#endif
  }
}

namespace ms_using_declaration_bug {

class A {
public: 
  int f(); 
};

class B : public A {
private:   
  using A::f;
  void g() {
    f(); // no diagnostic
  }
};

class C : public B { 
private:   
  using B::f; // expected-warning {{using declaration referring to inaccessible member 'ms_using_declaration_bug::B::f' (which refers to accessible member 'ms_using_declaration_bug::A::f') is a Microsoft compatibility extension}}
};

}

namespace using_tag_redeclaration
{
  struct S;
  namespace N {
    using ::using_tag_redeclaration::S;
    struct S {}; // expected-note {{previous definition is here}}
  }
  void f() {
    N::S s1;
    S s2;
  }
  void g() {
    struct S; // expected-note {{forward declaration of 'S'}}
    S s3; // expected-error {{variable has incomplete type 'S'}}
  }
  void h() {
    using ::using_tag_redeclaration::S;
    struct S {}; // expected-error {{redefinition of 'S'}}
  }
}


namespace MissingTypename {

template<class T> class A {
public:
	 typedef int TYPE;
};

template<class T> class B {
public:
	 typedef int TYPE;
};


template<class T, class U>
class C : private A<T>, public B<U> {
public:
   typedef A<T> Base1;
   typedef B<U> Base2;
   typedef A<U> Base3;

   A<T>::TYPE a1; // expected-warning {{missing 'typename' prior to dependent type name}}
   Base1::TYPE a2; // expected-warning {{missing 'typename' prior to dependent type name}}

   B<U>::TYPE a3; // expected-warning {{missing 'typename' prior to dependent type name}}
   Base2::TYPE a4; // expected-warning {{missing 'typename' prior to dependent type name}}

   A<U>::TYPE a5; // expected-error {{missing 'typename' prior to dependent type name}}
   Base3::TYPE a6; // expected-error {{missing 'typename' prior to dependent type name}}
 };

class D {
public:
    typedef int Type;
};

template <class T>
void function_missing_typename(const T::Type param)// expected-warning {{missing 'typename' prior to dependent type name}}
{
    const T::Type var = 2; // expected-warning {{missing 'typename' prior to dependent type name}}
}

template void function_missing_typename<D>(const D::Type param);

}

//MSVC allows forward enum declaration
enum ENUM; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
ENUM *var = 0;     
ENUM var2 = (ENUM)3;
enum ENUM1* var3 = 0;// expected-warning {{forward references to 'enum' types are a Microsoft extension}}

enum ENUM1 { kA };
enum ENUM1;  // This way round is fine.

enum ENUM2 {
	ENUM2_a = (enum ENUM2) 4,
	ENUM2_b = 0x9FFFFFFF, // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
	ENUM2_c = 0x100000000 // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
};

namespace NsEnumForwardDecl {
  enum E *p; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
  extern E e;
}
// Clang used to complain that NsEnumForwardDecl::E was undeclared below.
NsEnumForwardDecl::E NsEnumForwardDecl_e;
namespace NsEnumForwardDecl {
  extern E e;
}

namespace PR11791 {
  template<class _Ty>
  void del(_Ty *_Ptr) {
    _Ptr->~_Ty();  // expected-warning {{pseudo-destructors on type void are a Microsoft extension}}
  }

  void f() {
    int* a = 0;
    del((void*)a);  // expected-note {{in instantiation of function template specialization}}
  }
}

namespace IntToNullPtrConv {
  struct Foo {
    static const int ZERO = 0;
    typedef void (Foo::*MemberFcnPtr)();
  };

  struct Bar {
    const Foo::MemberFcnPtr pB;
  };

  Bar g_bar = { (Foo::MemberFcnPtr)Foo::ZERO };

  template<int N> int *get_n() { return N; }   // expected-warning {{expression which evaluates to zero treated as a null pointer constant}}
  int *g_nullptr = get_n<0>();  // expected-note {{in instantiation of function template specialization}}
}

namespace signed_hex_i64 {
void f(long long);
void f(int);
void g() {
  // This is an ambiguous call in standard C++.
  // This calls f(long long) in Microsoft mode because LL is always signed.
  f(0xffffffffffffffffLL);
  f(0xffffffffffffffffi64);
}
}

typedef void (*FnPtrTy)();
void (*PR23733_1)() = static_cast<FnPtrTy>((void *)0); // expected-warning {{static_cast between pointer-to-function and pointer-to-object is a Microsoft extension}}
void (*PR23733_2)() = FnPtrTy((void *)0);
void (*PR23733_3)() = (FnPtrTy)((void *)0);
void (*PR23733_4)() = reinterpret_cast<FnPtrTy>((void *)0);

long function_prototype(int a);
long (*function_ptr)(int a);

void function_to_voidptr_conv() {
  void *a1 = function_prototype;  // expected-warning {{implicit conversion between pointer-to-function and pointer-to-object is a Microsoft extension}}
  void *a2 = &function_prototype; // expected-warning {{implicit conversion between pointer-to-function and pointer-to-object is a Microsoft extension}}
  void *a3 = function_ptr;        // expected-warning {{implicit conversion between pointer-to-function and pointer-to-object is a Microsoft extension}}
}
