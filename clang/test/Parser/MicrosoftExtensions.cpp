// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-extensions -fms-compatibility -fdelayed-template-parsing

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

struct __declspec(uuid(L"00000000-0000-0000-1234-000000000047")) uuid_attr_bad1 { };// expected-error {{'uuid' attribute requires parameter 1 to be a string}}
struct __declspec(uuid(3)) uuid_attr_bad2 { };// expected-error {{'uuid' attribute requires parameter 1 to be a string}}
struct __declspec(uuid("0000000-0000-0000-1234-0000500000047")) uuid_attr_bad3 { };// expected-error {{uuid attribute contains a malformed GUID}}
struct __declspec(uuid("0000000-0000-0000-Z234-000000000047")) uuid_attr_bad4 { };// expected-error {{uuid attribute contains a malformed GUID}}
struct __declspec(uuid("000000000000-0000-1234-000000000047")) uuid_attr_bad5 { };// expected-error {{uuid attribute contains a malformed GUID}}



struct __declspec(uuid("000000A0-0000-0000-C000-000000000046"))
struct_with_uuid { };
struct struct_without_uuid { };

struct __declspec(uuid("000000A0-0000-0000-C000-000000000049"))
struct_with_uuid2;

struct 
struct_with_uuid2 {} ;

int uuid_sema_test()
{
   struct_with_uuid var_with_uuid[1];
   struct_without_uuid var_without_uuid[1];

   __uuidof(struct_with_uuid);
   __uuidof(struct_with_uuid2);
   __uuidof(struct_without_uuid); // expected-error {{cannot call operator __uuidof on a type with no GUID}}
   __uuidof(struct_with_uuid*);
   __uuidof(struct_without_uuid*); // expected-error {{cannot call operator __uuidof on a type with no GUID}}

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


template <class T, const GUID* g = &__uuidof(T)>
class COM_CLASS_TEMPLATE  { };

typedef COM_CLASS_TEMPLATE<struct_with_uuid, &__uuidof(struct_with_uuid)> COM_TYPE_1;
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



class AAAA { };

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


__interface MicrosoftInterface;
__interface MicrosoftInterface {
   void foo1() = 0;
   virtual void foo2() = 0;
};

__interface MicrosoftDerivedInterface : public MicrosoftInterface {
  void foo1();
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


namespace If_exists_test {

class IF_EXISTS {
private:
    typedef int Type;
};

int __if_exists_test() {
  int b=0;
  __if_exists(IF_EXISTS::Type) {
     b++;
     b++;
  }
  __if_exists(IF_EXISTS::Type_not) {
     this wont compile.
  }
  __if_not_exists(IF_EXISTS::Type) {
     this wont compile.
  }
  __if_not_exists(IF_EXISTS::Type_not) {
     b++;
     b++;
  }
}


__if_exists(IF_EXISTS::Type) {
  int var23;
}

__if_exists(IF_EXISTS::Type_not) {
 this wont compile.
}

__if_not_exists(IF_EXISTS::Type) {
 this wont compile.
}

__if_not_exists(IF_EXISTS::Type_not) {
  int var244;
}

int __if_exists_init_list() {

  int array1[] = {
    0, 
    __if_exists(IF_EXISTS::Type) {2, }
    3
  };

  int array2[] = {
    0, 
    __if_exists(IF_EXISTS::Type_not) { this wont compile }
    3
  };

  int array3[] = {
    0, 
    __if_not_exists(IF_EXISTS::Type_not) {2, }
    3
  };

  int array4[] = {
    0, 
    __if_not_exists(IF_EXISTS::Type) { this wont compile }
    3
  };

}


class IF_EXISTS_CLASS_TEST {
  __if_exists(IF_EXISTS::Type) {
    // __if_exists, __if_not_exists can nest
    __if_not_exists(IF_EXISTS::Type_not) {
      int var123;
    }
    int var23;
  }

  __if_exists(IF_EXISTS::Type_not) {
   this wont compile.
  }

  __if_not_exists(IF_EXISTS::Type) {
   this wont compile.
  }

  __if_not_exists(IF_EXISTS::Type_not) {
    int var244;
  }
};

}


int __identifier(generic) = 3;

class inline_definition_pure_spec {
   virtual int f() = 0 { return 0; }// expected-warning {{function definition with pure-specifier is a Microsoft extension}}
   virtual int f2() = 0;
};


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
