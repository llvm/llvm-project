// RUN: %clang_cc1 %s -triple i386-pc-win32 -std=c++14 -fsyntax-only -Wno-unused-getter-return-value -Wno-unused-value -Wmicrosoft -verify -fms-extensions -fms-compatibility -fdelayed-template-parsing

/* Microsoft attribute tests */
[repeatable][source_annotation_attribute( Parameter|ReturnValue )]
struct SA_Post{ SA_Post(); int attr; };

[returnvalue:SA_Post( attr=1)]
int foo1([SA_Post(attr=1)] void *param);

namespace {
  [returnvalue:SA_Post(attr=1)]
  int foo2([SA_Post(attr=1)] void *param);
}

class T {
  [returnvalue:SA_Post(attr=1)]
  int foo3([SA_Post(attr=1)] void *param);
};

extern "C" {
  [returnvalue:SA_Post(attr=1)]
  int foo5([SA_Post(attr=1)] void *param);
}

class class_attr {
public:
  class_attr([SA_Pre(Null=SA_No,NullTerminated=SA_Yes)]  int a)
  {
  }
};



void uuidof_test1()
{
  __uuidof(0);  // expected-error {{you need to include <guiddef.h> before using the '__uuidof' operator}}
}

typedef struct _GUID
{
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;

struct __declspec(uuid(L"00000000-0000-0000-1234-000000000047")) uuid_attr_bad1 { };// expected-error {{'uuid' attribute requires a string}}
struct __declspec(uuid(3)) uuid_attr_bad2 { };// expected-error {{'uuid' attribute requires a string}}
struct __declspec(uuid("0000000-0000-0000-1234-0000500000047")) uuid_attr_bad3 { };// expected-error {{uuid attribute contains a malformed GUID}}
struct __declspec(uuid("0000000-0000-0000-Z234-000000000047")) uuid_attr_bad4 { };// expected-error {{uuid attribute contains a malformed GUID}}
struct __declspec(uuid("000000000000-0000-1234-000000000047")) uuid_attr_bad5 { };// expected-error {{uuid attribute contains a malformed GUID}}
[uuid("000000000000-0000-1234-000000000047")] struct uuid_attr_bad6 { };// expected-error {{uuid attribute contains a malformed GUID}}

__declspec(uuid("000000A0-0000-0000-C000-000000000046")) int i; // expected-warning {{'uuid' attribute only applies to structs, unions, classes, and enums}}

struct __declspec(uuid("000000A0-0000-0000-C000-000000000046"))
struct_with_uuid { };
struct struct_without_uuid { };

struct __declspec(uuid("000000A0-0000-0000-C000-000000000049"))
struct_with_uuid2;

[uuid("000000A0-0000-0000-C000-000000000049")] struct struct_with_uuid3; // expected-warning{{specifying 'uuid' as an ATL attribute is deprecated; use __declspec instead}}

struct
struct_with_uuid2 {} ;

enum __declspec(uuid("000000A0-0000-0000-C000-000000000046"))
enum_with_uuid { };
enum enum_without_uuid { };

int __declspec(uuid("000000A0-0000-0000-C000-000000000046")) inappropriate_uuid; // expected-warning {{'uuid' attribute only applies to}}

int uuid_sema_test()
{
   struct_with_uuid var_with_uuid[1];
   struct_without_uuid var_without_uuid[1];

   __uuidof(struct_with_uuid);
   __uuidof(struct_with_uuid2);
   __uuidof(struct_with_uuid3);
   __uuidof(struct_without_uuid); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(struct_with_uuid*);
   __uuidof(struct_without_uuid*); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(struct_with_uuid[1]);
   __uuidof(struct_with_uuid*[1]); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(const struct_with_uuid[1][1]);
   __uuidof(const struct_with_uuid*[1][1]); // expected-error {{cannot call operator __uuidof on a type with no GUID}}

   __uuidof(enum_with_uuid);
   __uuidof(enum_without_uuid); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(enum_with_uuid*);
   __uuidof(enum_without_uuid*); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(enum_with_uuid[1]);
   __uuidof(enum_with_uuid*[1]); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(const enum_with_uuid[1][1]);
   __uuidof(const enum_with_uuid*[1][1]); // expected-error {{cannot call operator __uuidof on a type with no GUID}}

   __uuidof(var_with_uuid);
   __uuidof(var_without_uuid);// expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(var_with_uuid[1]);
   __uuidof(var_without_uuid[1]);// expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(&var_with_uuid[1]);
   __uuidof(&var_without_uuid[1]);// expected-error {{cannot call operator __uuidof on a type with no GUID}}

   __uuidof(0);
   __uuidof(1);// expected-error {{cannot call operator __uuidof on a type with no GUID}}
}


template <class T>
void template_uuid()
{
   T expr;

   __uuidof(T);
   __uuidof(expr);
}


template <class T, const GUID* g = &__uuidof(T)> // expected-note {{template parameter is declared here}}
class COM_CLASS_TEMPLATE  { };

typedef COM_CLASS_TEMPLATE<struct_with_uuid, &*&__uuidof(struct_with_uuid)> COM_TYPE_1; // expected-warning {{non-type template argument containing a dereference operation is a Microsoft extension}}
typedef COM_CLASS_TEMPLATE<struct_with_uuid> COM_TYPE_2;

template <class T, const GUID& g>
class COM_CLASS_TEMPLATE_REF  { };
typedef COM_CLASS_TEMPLATE_REF<struct_with_uuid, __uuidof(struct_with_uuid)> COM_TYPE_REF;

  struct late_defined_uuid;
  template<typename T>
  void test_late_defined_uuid() {
    __uuidof(late_defined_uuid);
  }
  struct __declspec(uuid("000000A0-0000-0000-C000-000000000049")) late_defined_uuid;

COM_CLASS_TEMPLATE_REF<int, __uuidof(struct_with_uuid)> good_template_arg;

COM_CLASS_TEMPLATE<int, __uuidof(struct_with_uuid)> bad_template_arg; // expected-error {{non-type template argument of type 'const _GUID' is not a constant expression}}
// expected-note@-1 {{read of object '__uuidof(struct_with_uuid)' whose value is not known}}
// expected-note@-2 {{temporary created here}}

namespace PR16911 {
struct __declspec(uuid("{12345678-1234-1234-1234-1234567890aB}")) uuid;
struct __declspec(uuid("{12345678-1234-1234-1234-1234567890aB}")) uuid2;

template <typename T, typename T2>
struct thing {
};

struct empty {};
struct inher : public thing<empty, uuid2> {};

struct __declspec(uuid("{12345678-1234-1234-1234-1234567890aB}")) uuid;
const struct _GUID *w = &__uuidof(inher); // expected-error{{cannot call operator __uuidof on a type with no GUID}}
const struct _GUID *x = &__uuidof(thing<uuid, inher>);
const struct _GUID *y = &__uuidof(thing<uuid2, uuid>); // expected-error{{cannot call operator __uuidof on a type with multiple GUIDs}}
thing<uuid2, uuid> thing_obj = thing<uuid2, uuid>();
const struct _GUID *z = &__uuidof(thing_obj); // expected-error{{cannot call operator __uuidof on a type with multiple GUIDs}}
}

class CtorCall {
public:
  CtorCall& operator=(const CtorCall& that);

  int a;
};

CtorCall& CtorCall::operator=(const CtorCall& that)
{
    if (this != &that) {
        this->CtorCall::~CtorCall();
        this->CtorCall::CtorCall(that); // expected-warning {{explicit constructor calls are a Microsoft extension}}
    }
    return *this;
}

template <class A>
class C1 {
public:
  template <int B>
  class Iterator {
  };
};

template<class T>
class C2  {
  typename C1<T>:: /*template*/  Iterator<0> Mypos; // expected-warning {{use 'template' keyword to treat 'Iterator' as a dependent template name}}
};

template <class T>
void missing_template_keyword(){
  typename C1<T>:: /*template*/ Iterator<0> Mypos; // expected-warning {{use 'template' keyword to treat 'Iterator' as a dependent template name}}
}



class AAAA {
   typedef int D;
};

template <typename T>
class SimpleTemplate {};

template <class T>
void redundant_typename() {
   typename T t;// expected-warning {{expected a qualified name after 'typename'}}
   typename AAAA a;// expected-warning {{expected a qualified name after 'typename'}}

   t = 3;

   typedef typename T* pointerT;// expected-warning {{expected a qualified name after 'typename'}}
   typedef typename SimpleTemplate<int> templateT;// expected-warning {{expected a qualified name after 'typename'}}

   pointerT pT = &t;
   *pT = 4;

   int var;
   int k = typename var;// expected-error {{expected a qualified name after 'typename'}}
}

template <typename T>
struct TypenameWrongPlace {
  typename typedef T::D D;// expected-warning {{expected a qualified name after 'typename'}}
};

extern TypenameWrongPlace<AAAA> PR16925;

__interface MicrosoftInterface;
__interface MicrosoftInterface {
   void foo1() = 0; // expected-note {{overridden virtual function is here}}
   virtual void foo2() = 0;
};

__interface MicrosoftDerivedInterface : public MicrosoftInterface {
  void foo1(); // expected-warning {{'foo1' overrides a member function but is not marked 'override'}}
  void foo2() override;
  void foo3();
};

void interface_test() {
  MicrosoftInterface* a;
  a->foo1();
  MicrosoftDerivedInterface* b;
  b->foo2();
}

__int64 x7 = __int64(0);
_int64 x8 = _int64(0);
static_assert(sizeof(_int64) == 8, "");
static_assert(sizeof(_int32) == 4, "");
static_assert(sizeof(_int16) == 2, "");
static_assert(sizeof(_int8) == 1, "");

int __identifier(generic) = 3;
int __identifier(int) = 4;
struct __identifier(class) { __identifier(class) *__identifier(for); };
__identifier(class) __identifier(struct) = { &__identifier(struct) };

int __identifier for; // expected-error {{missing '(' after '__identifier'}}
int __identifier(else} = __identifier(for); // expected-error {{missing ')' after identifier}} expected-note {{to match this '('}}
#define identifier_weird(x) __identifier(x
int k = identifier_weird(if)); // expected-error {{use of undeclared identifier 'if'}}

extern int __identifier(and);

void f() {
  __identifier(() // expected-error {{cannot convert '(' token to an identifier}}
  __identifier(void) // expected-error {{use of undeclared identifier 'void'}}
  __identifier()) // expected-error {{cannot convert ')' token to an identifier}}
  // FIXME: We should pick a friendlier display name for this token kind.
  __identifier(1) // expected-error {{cannot convert <numeric_constant> token to an identifier}}
  __identifier(+) // expected-error {{cannot convert '+' token to an identifier}}
  __identifier("foo") // expected-error {{cannot convert <string_literal> token to an identifier}}
  __identifier(;) // expected-error {{cannot convert ';' token to an identifier}}
}

class inline_definition_pure_spec {
   virtual int f() = 0 { return 0; }// expected-warning {{function definition with pure-specifier is a Microsoft extension}}
   virtual int f2() = 0;
};

struct pure_virtual_dtor {
  virtual ~pure_virtual_dtor() = 0;
};
pure_virtual_dtor::~pure_virtual_dtor() { }

struct pure_virtual_dtor_inline {
  virtual ~pure_virtual_dtor_inline() = 0 { }// expected-warning {{function definition with pure-specifier is a Microsoft extension}}
};

template<typename T> struct pure_virtual_dtor_template {
  virtual ~pure_virtual_dtor_template() = 0;
};
template<typename T> pure_virtual_dtor_template<T>::~pure_virtual_dtor_template() {}
template struct pure_virtual_dtor_template<int>;

template<typename T> struct pure_virtual_dtor_template_inline {
    virtual ~pure_virtual_dtor_template_inline() = 0 {}
    // expected-warning@-1 2{{function definition with pure-specifier is a Microsoft extension}}
};
template struct pure_virtual_dtor_template_inline<int>;
// expected-note@-1 {{in instantiation of member function}}

int main () {
  // Necessary to force instantiation in -fdelayed-template-parsing mode.
  test_late_defined_uuid<int>();
  redundant_typename<int>();
  missing_template_keyword<int>();
}

namespace access_protected_PTM {
  class A {
  protected:
    void f(); // expected-note {{must name member using the type of the current context 'access_protected_PTM::B'}}
  };

  class B : public A{
  public:
    void test_access();
    static void test_access_static();
  };

  void B::test_access() {
    &A::f; // expected-error {{'f' is a protected member of 'access_protected_PTM::A'}}
  }

  void B::test_access_static() {
    &A::f;
  }
}

namespace Inheritance {
  class __single_inheritance A;
  class __multiple_inheritance B;
  class __virtual_inheritance C;
}

struct StructWithProperty {
  __declspec(property) int V0; // expected-error {{expected '(' after 'property'}}
  __declspec(property()) int V1; // expected-error {{property does not specify a getter or a putter}}
  __declspec(property(set)) int V2; // expected-error {{putter for property must be specified as 'put', not 'set'}} expected-error {{expected '=' after 'set'}}
  __declspec(property(ptu)) int V3; // expected-error {{missing 'get=' or 'put='}}
  __declspec(property(ptu=PutV)) int V4; // expected-error {{expected 'get' or 'put' in property declaration}}
  __declspec(property(get)) int V5; // expected-error {{expected '=' after 'get'}}
  __declspec(property(get&)) int V6; // expected-error {{expected '=' after 'get'}}
  __declspec(property(get=)) int V7; // expected-error {{expected name of accessor method}}
  __declspec(property(get=GetV)) int V8; // no-warning
  __declspec(property(get=GetV=)) int V9; // expected-error {{expected ',' or ')' at end of property accessor list}}
  __declspec(property(get=GetV,)) int V10; // expected-error {{expected 'get' or 'put' in property declaration}}
  __declspec(property(get=GetV,put=SetV)) int V11; // no-warning
  __declspec(property(get=GetV,put=SetV,get=GetV)) int V12; // expected-error {{property declaration specifies 'get' accessor twice}}
  __declspec(property(get=GetV)) int V13 = 3; // expected-error {{property declaration cannot have an in-class initializer}}

  int GetV() { return 123; }
  void SetV(int v) {}
};
void TestProperty() {
  StructWithProperty sp;
  sp.V8;
  sp.V8 = 0; // expected-error {{no setter defined for property 'V8'}}
  int i = sp.V11;
  sp.V11 = i++;
  sp.V11 += 8;
  sp.V11++;
  ++sp.V11;
}

//expected-warning@+1 {{C++ operator 'and' (aka '&&') used as a macro name}}
#define and foo

struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) __declspec(novtable) IUnknown {};

typedef bool (__stdcall __stdcall *blarg)(int);

void local_callconv() {
  bool (__stdcall *p)(int);
}

struct S7 {
	int foo() { return 12; }
	__declspec(property(get=foo) deprecated) int t; // expected-note {{'t' has been explicitly marked deprecated here}}
};

// Technically, this is legal (though it does nothing)
__declspec() void quux( void ) {
  struct S7 s;
  int i = s.t;	// expected-warning {{'t' is deprecated}}
}

void *_alloca(int);

void foo(void) {
  __declspec(align(16)) int *buffer = (int *)_alloca(9);
}

template <int *>
struct NullptrArg {};
NullptrArg<nullptr> a;

// Ignored type qualifiers after comma in declarator lists
typedef int ignored_quals_dummy1, const volatile __ptr32 __ptr64 __w64 __unaligned __sptr __uptr ignored_quals1; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy2)(), __fastcall ignored_quals2; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy3)(), __stdcall ignored_quals3; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy4)(), __thiscall ignored_quals4; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy5)(), __cdecl ignored_quals5; // expected-warning {{qualifiers after comma in declarator list are ignored}}
typedef void(*ignored_quals_dummy6)(), __vectorcall ignored_quals6; // expected-warning {{qualifiers after comma in declarator list are ignored}}

namespace {
bool f(int);
template <typename T>
struct A {
  constexpr A(T t) {
    __assume(f(t)); // expected-warning{{the argument to '__assume' has side effects that will be discarded}}
  }
  constexpr bool g() { return false; }
};
constexpr A<int> h() {
  A<int> b(0); // expected-note {{in instantiation of member function}}
  return b;
}
static_assert(h().g() == false, "");
}

namespace {
__declspec(align(16)) struct align_before_key1 {};
__declspec(align(16)) struct align_before_key2 {} align_before_key2_var;
__declspec(align(16)) struct align_before_key3 {} *align_before_key3_var;
static_assert(__alignof(struct align_before_key1) == 16, "");
static_assert(__alignof(struct align_before_key2) == 16, "");
static_assert(__alignof(struct align_before_key3) == 16, "");
}

namespace PR24027 {
struct S {
  template <typename T>
  S(T);
} f([] {});
}

namespace pr36638 {
// Make sure we accept __unaligned method qualifiers on member function
// pointers.
struct A;
void (A::*mp1)(int) __unaligned;
}
