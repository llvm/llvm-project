// RUN: %clang_cc1 -fspell-checking-limit 0 -verify -Wno-c++11-extensions %s
// RUN: %clang_cc1 -fspell-checking-limit 0 -verify -Wno-c++11-extensions -std=c++20 %s

namespace PR21817{
int a(-rsing[2]); // expected-error {{undeclared identifier 'rsing'; did you mean 'using'?}}
                  // expected-error@-1 {{expected expression}}
}

struct errc {
  int v_;
  operator int() const {return v_;}
};

class error_condition
{
  int _val_;
public:
  error_condition() : _val_(0) {}

  error_condition(int _val)
    : _val_(_val) {}

  template <class E>
  error_condition(E _e) {
    // make_error_condition must not be typo corrected to error_condition
    // even though the first declaration of make_error_condition has not
    // yet been encountered. This was a bug in the first version of the type
    // name typo correction patch that wasn't noticed until building LLVM with
    // Clang failed.
    *this = make_error_condition(_e);
  }

};

inline error_condition make_error_condition(errc _e) {
  return error_condition(static_cast<int>(_e));
}


// Prior to the introduction of a callback object to further filter possible
// typo corrections, this example would not trigger a suggestion as "base_type"
// is a closer match to "basetype" than is "BaseType" but "base_type" does not
// refer to a base class or non-static data member.
struct BaseType { };
struct Derived : public BaseType { // expected-note {{base class 'BaseType' specified here}}
  static int base_type; // expected-note {{'base_type' declared here}}
  Derived() : basetype() {} // expected-error{{initializer 'basetype' does not name a non-static data member or base class; did you mean the base class 'BaseType'?}}
};

// Test the improvement from passing a callback object to CorrectTypo in
// the helper function LookupMemberExprInRecord.
int get_type(struct Derived *st) {
  return st->Base_Type; // expected-error{{no member named 'Base_Type' in 'Derived'; did you mean 'base_type'?}}
}

// In this example, somename should not be corrected to the cached correction
// "some_name" since "some_name" is a class and a namespace name is needed.
class some_name {}; // expected-note {{'some_name' declared here}}
somename Foo; // expected-error {{unknown type name 'somename'; did you mean 'some_name'?}}
namespace SomeName {} // expected-note {{namespace 'SomeName' defined here}}
using namespace somename; // expected-error {{no namespace named 'somename'; did you mean 'SomeName'?}}


// Without the callback object, CorrectTypo would choose "field1" as the
// correction for "fielda" as it is closer than "FieldA", but that correction
// would be later discarded by the caller and no suggestion would be given.
struct st {
  struct {
    int field1;
  };
  double FieldA; // expected-note{{'FieldA' declared here}}
};
st var = { .fielda = 0.0 }; // expected-error{{field designator 'fielda' does not refer to any field in type 'st'; did you mean 'FieldA'?}}

// Test the improvement from passing a callback object to CorrectTypo in
// Sema::BuildCXXNestedNameSpecifier. And also for the improvement by doing
// so in Sema::getTypeName.
typedef char* another_str; // expected-note{{'another_str' declared here}}
namespace AnotherStd { // expected-note{{'AnotherStd' declared here}}
  class string {};
}
another_std::string str; // expected-error{{use of undeclared identifier 'another_std'; did you mean 'AnotherStd'?}}
another_str *cstr = new AnotherStr; // expected-error{{unknown type name 'AnotherStr'; did you mean 'another_str'?}}

// Test the improvement from passing a callback object to CorrectTypo in
// Sema::ActOnSizeofParameterPackExpr.
char* TireNames;
template<typename ...TypeNames> struct count { // expected-note{{parameter pack 'TypeNames' declared here}}
  static const unsigned value = sizeof...(TyreNames); // expected-error{{'TyreNames' does not refer to the name of a parameter pack; did you mean 'TypeNames'?}}
};

// Test the typo-correction callback in Sema::DiagnoseUnknownTypeName.
namespace unknown_type_test {
  class StreamOut {}; // expected-note 2 {{'StreamOut' declared here}}
  long stream_count; // expected-note 2 {{'stream_count' declared here}}
};
unknown_type_test::stream_out out; // expected-error{{no type named 'stream_out' in namespace 'unknown_type_test'; did you mean 'StreamOut'?}}

// Demonstrate a case where using only the cached value returns the wrong thing
// when the cached value was the result of a previous callback object that only
// accepts a subset of the current callback object.
namespace cache_invalidation_test {
using namespace unknown_type_test;
void bar(long i);
void before_caching_classname() {
  bar((stream_out)); // expected-error{{use of undeclared identifier 'stream_out'; did you mean 'stream_count'?}}
}
stream_out out; // expected-error{{unknown type name 'stream_out'; did you mean 'StreamOut'?}}
void after_caching_classname() {
  bar((stream_out)); // expected-error{{use of undeclared identifier 'stream_out'; did you mean 'stream_count'?}}
}
}

// Test the typo-correction callback in Sema::DiagnoseInvalidRedeclaration.
struct BaseDecl {
  void add_in(int i);
};
struct TestRedecl : public BaseDecl {
  void add_it(int i); // expected-note{{'add_it' declared here}}
};
void TestRedecl::add_in(int i) {} // expected-error{{out-of-line definition of 'add_in' does not match any declaration in 'TestRedecl'; did you mean 'add_it'?}}

// Test the improved typo correction for the Parser::ParseCastExpr =>
// Sema::ActOnIdExpression => Sema::DiagnoseEmptyLookup call path.
class SomeNetMessage; // expected-note 2{{'SomeNetMessage'}}
class Message {};
void foo(Message&);
void foo(SomeNetMessage&);
void doit(void *data) {
  Message somenetmsg; // expected-note{{'somenetmsg' declared here}}
  foo(somenetmessage); // expected-error{{use of undeclared identifier 'somenetmessage'; did you mean 'somenetmsg'?}}
  foo((somenetmessage)data); // expected-error{{unknown type name 'somenetmessage'; did you mean 'SomeNetMessage'?}} expected-error{{incomplete type}}
}

// Test the typo-correction callback in BuildRecoveryCallExpr.
// Solves the main issue in PR 9320 of suggesting corrections that take the
// wrong number of arguments.
void revoke(const char*); // expected-note 2{{'revoke' declared here}}
void Test() {
  Invoke(); // expected-error{{use of undeclared identifier 'Invoke'}}
  Invoke("foo"); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'revoke'?}}
  Invoke("foo", "bar"); // expected-error{{use of undeclared identifier 'Invoke'}}
}
void Test2(void (*invoke)(const char *, int)) { // expected-note{{'invoke' declared here}}
  Invoke(); // expected-error{{use of undeclared identifier 'Invoke'}}
  Invoke("foo"); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'revoke'?}}
  Invoke("foo", 7); // expected-error{{use of undeclared identifier 'Invoke'; did you mean 'invoke'?}}
  Invoke("foo", 7, 22); // expected-error{{use of undeclared identifier 'Invoke'}}
}

void provoke(const char *x, bool y=false) {} // expected-note 2{{'provoke' declared here}}
void Test3() {
  Provoke(); // expected-error{{use of undeclared identifier 'Provoke'}}
  Provoke("foo"); // expected-error{{use of undeclared identifier 'Provoke'; did you mean 'provoke'?}}
  Provoke("foo", true); // expected-error{{use of undeclared identifier 'Provoke'; did you mean 'provoke'?}}
  Provoke("foo", 7, 22); // expected-error{{use of undeclared identifier 'Provoke'}}
}

// PR 11737 - Don't try to typo-correct the implicit 'begin' and 'end' in a
// C++11 for-range statement.
struct R {};
bool begun(R);
void RangeTest() {
  for (auto b : R()) {} // expected-error {{invalid range expression of type 'R'}}
}

// PR 12019 - Avoid infinite mutual recursion in DiagnoseInvalidRedeclaration
// by not trying to typo-correct a method redeclaration to declarations not
// in the current record.
class Parent {
 void set_types(int index, int value);
 void add_types(int value);
};
class Child: public Parent {};
void Child::add_types(int value) {} // expected-error{{out-of-line definition of 'add_types' does not match any declaration in 'Child'}}

// Fix the callback based filtering of typo corrections within
// Sema::ActOnIdExpression by Parser::ParseCastExpression to allow type names as
// potential corrections for template arguments.
namespace clash {
class ConstructExpr {}; // expected-note 2{{'clash::ConstructExpr' declared here}}
}
class ClashTool {
  bool HaveConstructExpr();
  template <class T> T* getExprAs();

  void test() {
    ConstructExpr *expr = // expected-error{{unknown type name 'ConstructExpr'; did you mean 'clash::ConstructExpr'?}}
        getExprAs<ConstructExpr>(); // expected-error{{unknown type name 'ConstructExpr'; did you mean 'clash::ConstructExpr'?}}
  }
};

namespace test1 {
  struct S {
    struct Foobar *f;  // expected-note{{'Foobar' declared here}}
  };
  test1::FooBar *b;  // expected-error{{no type named 'FooBar' in namespace 'test1'; did you mean 'Foobar'?}}
}

namespace ImplicitInt {
  void f(int, unsinged); // expected-error{{did you mean 'unsigned'}}
  struct S {
    unsinged : 4; // expected-error{{did you mean 'unsigned'}}
  };
}

namespace PR13051 {
  template<typename T> struct S {
    template<typename U> void f();
    operator bool() const;
  };

  void foo(); // expected-note{{'foo' declared here}}
  void g(void(*)()); // expected-note{{candidate function not viable}}
  void g(bool(S<int>::*)() const); // expected-note{{candidate function not viable}}

  void test() {
    g(&S<int>::tempalte f<int>); // expected-error{{did you mean 'template'?}} \
                                 // expected-error{{no matching function for call to 'g'}}
    g(&S<int>::opeartor bool); // expected-error{{did you mean 'operator'?}}
    g(&S<int>::foo); // expected-error{{no member named 'foo' in 'PR13051::S<int>'; did you mean simply 'foo'?}}
  }
}

inf f(doulbe); // expected-error{{'int'}} expected-error{{'double'}}

namespace PR6325 {
class foo { }; // expected-note{{'foo' declared here}}
// Note that for this example (pulled from the PR), if keywords are not excluded
// as correction candidates then no suggestion would be given; correcting
// 'boo' to 'bool' is the same edit distance as correcting 'boo' to 'foo'.
class bar : boo { }; // expected-error{{unknown class name 'boo'; did you mean 'foo'?}}
}

namespace outer {
  void somefunc();  // expected-note{{'::outer::somefunc' declared here}}
  void somefunc(int, int);  // expected-note{{'::outer::somefunc' declared here}}

  namespace inner {
    void somefunc(int) {
      someFunc();  // expected-error{{use of undeclared identifier 'someFunc'; did you mean '::outer::somefunc'?}}
      someFunc(1, 2);  // expected-error{{use of undeclared identifier 'someFunc'; did you mean '::outer::somefunc'?}}
    }
  }
}

namespace b6956809_test1 {
  struct A {};
  struct B {};

  struct S1 {
    void method(A*);  // no note here
    void method(B*);  // expected-note{{'method' declared here}}
  };

  void test1() {
    B b;
    S1 s;
    s.methodd(&b);  // expected-error{{no member named 'methodd' in 'b6956809_test1::S1'; did you mean 'method'}}
  }

  struct S2 {
    S2();
    void method(A*) const;
   private:
    void method(B*);
  };

  void test2() {
    B b;
    const S2 s;
    s.methodd(&b);  // expected-error-re{{no member named 'methodd' in 'b6956809_test1::S2'{{$}}}}
  }
}

namespace b6956809_test2 {
  template<typename T> struct Err { typename T::error n; };  // expected-error{{type 'void *' cannot be used prior to '::' because it has no members}}
  struct S {
    template<typename T> typename Err<T>::type method(T);  // expected-note{{in instantiation of template class 'b6956809_test2::Err<void *>' requested here}}
    template<typename T> int method(T *);  // expected-note{{'method' declared here}}
  };

  void test() {
    S s;
    int k = s.methodd((void*)0);  // expected-error{{no member named 'methodd' in 'b6956809_test2::S'; did you mean 'method'?}} expected-note{{while substituting deduced template arguments into function template 'method' [with T = void *]}}
  }
}

namespace PR12951 {
// If there are two corrections that have the same identifier and edit distance
// and only differ by their namespaces, don't suggest either as a correction
// since both are equally likely corrections.
namespace foobar { struct Thing {}; }
namespace bazquux { struct Thing {}; }
void f() { Thing t; } // expected-error{{unknown type name 'Thing'}}
}

namespace bogus_keyword_suggestion {
void test() {
   status = "OK";  // expected-error-re {{use of undeclared identifier 'status'{{$}}}}
   return status;  // expected-error-re {{use of undeclared identifier 'status'{{$}}}}
 }
}

namespace PR13387 {
struct A {
  void CreateFoo(float, float);
  void CreateBar(float, float);
};
struct B : A {
  using A::CreateFoo; // expected-note {{'CreateFoo' declared here}}
  void CreateFoo(int, int);  // expected-note {{'CreateFoo' declared here}}
};
void f(B &x) {
  x.Createfoo(0,0);  // expected-error {{no member named 'Createfoo' in 'PR13387::B'; did you mean 'CreateFoo'?}}
  x.Createfoo(0.f,0.f);  // expected-error {{no member named 'Createfoo' in 'PR13387::B'; did you mean 'CreateFoo'?}}
}
}

namespace using_decl {
  namespace somewhere { int foobar; }
  using somewhere::foobar; // expected-note {{declared here}}
  int k = goobar; // expected-error {{did you mean 'foobar'?}}
}

struct DataStruct {void foo();};
struct T {
 DataStruct data_struct;
 void f();
};
// should be void T::f();
void f() {
 data_struct->foo();  // expected-error-re{{use of undeclared identifier 'data_struct'{{$}}}}
}

namespace PR12287 {
class zif {
  void nab(int);
};
void nab();  // expected-note{{'::PR12287::nab' declared here}}
void zif::nab(int) {
  nab();  // expected-error{{too few arguments to function call, expected 1, have 0; did you mean '::PR12287::nab'?}}
}
}

namespace TemplateFunction {
template <class T>
void fnA(T) { }  // expected-note {{'::TemplateFunction::fnA' declared here}}

template <class T>
void fnB(T) { }  // expected-note {{'::TemplateFunction::fnB' declared here}}

class Foo {
 public:
  void fnA(int, int) {}
  void fnB() {}
};

void test(Foo F, int num) {
  F.fnA(num);  // expected-error {{too few arguments to function call, expected 2, have 1; did you mean '::TemplateFunction::fnA'?}}
  F.fnB(num);  // expected-error {{too many arguments to function call, expected 0, have 1; did you mean '::TemplateFunction::fnB'?}}
}
}
namespace using_suggestion_val_dropped_specifier {
void FFF() {} // expected-note {{'::using_suggestion_val_dropped_specifier::FFF' declared here}}
namespace N { }
using N::FFF; // expected-error {{no member named 'FFF' in namespace 'using_suggestion_val_dropped_specifier::N'; did you mean '::using_suggestion_val_dropped_specifier::FFF'?}}
}

namespace class_member_typo_corrections {
class Outer {
public:
  class Inner {};  // expected-note {{'Outer::Inner' declared here}}
  Inner MyMethod(Inner arg);
};

Inner Outer::MyMethod(Inner arg) {  // expected-error {{unknown type name 'Inner'; did you mean 'Outer::Inner'?}}
  return Inner();
}

class Result {
public:
  enum ResultType {
    ENTITY,  // expected-note {{'Result::ENTITY' declared here}}
    PREDICATE,  // expected-note {{'Result::PREDICATE' declared here}}
    LITERAL  // expected-note {{'Result::LITERAL' declared here}}
  };

  ResultType type();
};

void test() {
  Result result_cell;
  switch (result_cell.type()) {
  case ENTITY:  // expected-error {{use of undeclared identifier 'ENTITY'; did you mean 'Result::ENTITY'?}}
  case LITERAL:  // expected-error {{use of undeclared identifier 'LITERAL'; did you mean 'Result::LITERAL'?}}
  case PREDICAT:  // expected-error {{use of undeclared identifier 'PREDICAT'; did you mean 'Result::PREDICATE'?}}
    break;
  }
}

class Figure {
  enum ResultType {
    SQUARE,
    TRIANGLE,
    CIRCLE
  };

public:
  ResultType type();
};

void testAccess() {
  Figure obj;
  switch (obj.type()) {
  case SQUARE:  // expected-error-re {{use of undeclared identifier 'SQUARE'{{$}}}}
  case TRIANGLE:  // expected-error-re {{use of undeclared identifier 'TRIANGLE'{{$}}}}
  case CIRCE:  // expected-error-re {{use of undeclared identifier 'CIRCE'{{$}}}}
    break;
  }
}
}

long readline(const char *, char *, unsigned long);
void assign_to_unknown_var() {
    deadline_ = 1;  // expected-error-re {{use of undeclared identifier 'deadline_'{{$}}}}
}

namespace no_ns_before_dot {
namespace re2 {}
void test() {
    req.set_check(false);  // expected-error-re {{use of undeclared identifier 'req'{{$}}}}
}
}

namespace PR17394 {
  class A {
  protected:
    long zzzzzzzzzz;
  };
  class B : private A {};
  B zzzzzzzzzy<>; // expected-error {{template specialization requires 'template<>'}} expected-error {{no variable template matches specialization}}
}

namespace correct_fields_in_member_funcs {
struct S {
  int my_member;  // expected-note {{'my_member' declared here}}
  void f() { my_menber = 1; }  // expected-error {{use of undeclared identifier 'my_menber'; did you mean 'my_member'?}}
};
}

namespace PR17019 {
  template<class F>
  struct evil {
    evil(F de) {  // expected-note {{'de' declared here}}
      de_;  // expected-error {{use of undeclared identifier 'de_'; did you mean 'de'?}} \
            // expected-warning {{expression result unused}}
    }
    ~evil() {
      de_->bar()  // expected-error {{use of undeclared identifier 'de_'}}
    }
  };

  void meow() {
    evil<int> Q(0); // expected-note {{in instantiation of member function}}
  }
}

namespace fix_class_name_qualifier {
class MessageHeaders {};
class MessageUtils {
 public:
  static void ParseMessageHeaders(int, int); // expected-note {{'MessageUtils::ParseMessageHeaders' declared here}}
};

void test() {
  // No, we didn't mean to call MessageHeaders::MessageHeaders.
  MessageHeaders::ParseMessageHeaders(5, 4); // expected-error {{no member named 'ParseMessageHeaders' in 'fix_class_name_qualifier::MessageHeaders'; did you mean 'MessageUtils::ParseMessageHeaders'?}}
}
}

namespace PR18213 {  // expected-note {{'PR18213' declared here}}
struct WrapperInfo {
  int i;
};

template <typename T> struct Wrappable {
  static WrapperInfo kWrapperInfo;
};

// Note the space before "::PR18213" is intended and needed, as it highlights
// the actual typo, which is the leading "::".
// TODO: Suggest removing the "::" from "::PR18213" (the right correction)
// instead of incorrectly suggesting dropping "PR18213::WrapperInfo::".
template <>
PR18213::WrapperInfo ::PR18213::Wrappable<int>::kWrapperInfo = { 0 };  // expected-error {{no member named 'PR18213' in 'PR18213::WrapperInfo'; did you mean simply 'PR18213'?}} \
                                                                       // expected-error {{C++ requires a type specifier for all declarations}}
}

namespace PR18651 {
struct {
  int x;
} a, b;

int y = x;  // expected-error-re {{use of undeclared identifier 'x'{{$}}}}
}

namespace PR18685 {
template <class C, int I, int J>
class SetVector {
 public:
  SetVector() {}
};

template <class C, int I>
class SmallSetVector : public SetVector<C, I, 8> {};

class foo {};
SmallSetVector<foo*, 2> fooSet;
}

PR18685::BitVector Map;  // expected-error-re {{no type named 'BitVector' in namespace 'PR18685'{{$}}}}

namespace shadowed_template {
template <typename T> class Fizbin {};  // expected-note {{'::shadowed_template::Fizbin' declared here}}
class Baz {
   int Fizbin;
   Fizbin<int> qux; // expected-error {{no template named 'Fizbin'; did you mean '::shadowed_template::Fizbin'?}}
};
}

namespace no_correct_template_id_to_non_template {
  struct Frobnatz {}; // expected-note {{declared here}}
  Frobnats fn; // expected-error {{unknown type name 'Frobnats'; did you mean 'Frobnatz'?}}
  Frobnats<int> fni; // expected-error-re {{no template named 'Frobnats'{{$}}}}
}

namespace PR18852 {
void func() {
  struct foo {
    void barberry() {}
  };
  barberry();  // expected-error-re {{use of undeclared identifier 'barberry'{{$}}}}
}

class Thread {
 public:
  void Start();
  static void Stop();  // expected-note {{'Thread::Stop' declared here}}
};

class Manager {
 public:
  void Start(int);  // expected-note {{'Start' declared here}}
  void Stop(int);  // expected-note {{'Stop' declared here}}
};

void test(Manager *m) {
  // Don't suggest Thread::Start as a correction just because it has the same
  // (unqualified) name and accepts the right number of args; this is a method
  // call on an object in an unrelated class.
  m->Start();  // expected-error-re {{too few arguments to function call, expected 1, have 0{{$}}}}
  m->Stop();  // expected-error-re {{too few arguments to function call, expected 1, have 0{{$}}}}
  Stop();  // expected-error {{use of undeclared identifier 'Stop'; did you mean 'Thread::Stop'?}}
}

}

namespace std {
class bernoulli_distribution {
 public:
  double p() const;
};
}
void test() {
  // Make sure that typo correction doesn't suggest changing 'p' to
  // 'std::bernoulli_distribution::p' as that is most likely wrong.
  if (p)  // expected-error-re {{use of undeclared identifier 'p'{{$}}}}
    return;
}

namespace PR19681 {
  struct TypoA {};
  struct TypoB {
    void test();
  private:
    template<typename T> void private_memfn(T);  // expected-note{{declared here}}
  };
  void TypoB::test() {
    // FIXME: should suggest 'PR19681::TypoB::private_memfn' instead of '::PR19681::TypoB::private_memfn'
    (void)static_cast<void(TypoB::*)(int)>(&TypoA::private_memfn);  // expected-error{{no member named 'private_memfn' in 'PR19681::TypoA'; did you mean '::PR19681::TypoB::private_memfn'?}}
  }
}

namespace testWantFunctionLikeCasts {
  long test(bool a) {
    if (a)
      return struc(5.7);  // expected-error-re {{use of undeclared identifier 'struc'{{$}}}}
    else
      return lon(8.0);  // expected-error {{use of undeclared identifier 'lon'; did you mean 'long'?}}
  }
}

namespace testCXXDeclarationSpecifierParsing {
namespace test {
  struct SomeSettings {};  // expected-note {{'test::SomeSettings' declared here}}
}
class Test {};
int bar() {
  Test::SomeSettings some_settings; // expected-error {{no type named 'SomeSettings' in 'testCXXDeclarationSpecifierParsing::Test'; did you mean 'test::SomeSettings'?}}
}
}

namespace testIncludeTypeInTemplateArgument {
template <typename T, typename U>
void foo(T t = {}, U = {}); // expected-note {{candidate template ignored}}

class AddObservation {}; // expected-note {{declared here}}
int bar1() {
  // should resolve to a class.
  foo<AddObservationFn, int>(); // expected-error {{unknown type name 'AddObservationFn'; did you mean 'AddObservation'?}}

  // should not resolve to a class.
  foo(AddObservationFn, 1);    // expected-error-re {{use of undeclared identifier 'AddObservationFn'{{$}}}}
  int a = AddObservationFn, b; // expected-error-re {{use of undeclared identifier 'AddObservationFn'{{$}}}}

  int AddObservation; // expected-note 3{{declared here}}
  // should resolve to a local variable.
  foo(AddObservationFn, 1);    // expected-error {{use of undeclared identifier 'AddObservationFn'; did you mean}}
  int c = AddObservationFn, d; // expected-error {{use of undeclared identifier 'AddObservationFn'; did you mean}}

  // FIXME: would be nice to not resolve to a variable.
  foo<AddObservationFn, int>(); // expected-error {{use of undeclared identifier 'AddObservationFn'; did you mean}} \
                                   expected-error {{no matching function for call}}
}
} // namespace testIncludeTypeInTemplateArgument

namespace testNoCrashOnNullNNSTypoCorrection {
int AddObservation();
template <typename T, typename... Args>
class UsingImpl {};
class AddObservation { // expected-note {{declared here}}
  using Using =
      // should resolve to a class.
      UsingImpl<AddObservationFn, const int>; // expected-error {{unknown type name 'AddObservationFn'; did you mean}}
};
} // namespace testNoCrashOnNullNNSTypoCorrection

namespace testNonStaticMemberHandling {
struct Foo {
  bool usesMetadata;  // expected-note {{'usesMetadata' declared here}}
};
int test(Foo f) {
  if (UsesMetadata)  // expected-error-re {{use of undeclared identifier 'UsesMetadata'{{$}}}}
    return 5;
  if (f.UsesMetadata)  // expected-error {{no member named 'UsesMetadata' in 'testNonStaticMemberHandling::Foo'; did you mean 'usesMetadata'?}}
    return 11;
  return 0;
}
};

namespace testMemberExprDeclarationNameInfo {
  // The AST should only have the corrected name with no mention of 'data_'.
  void f(int);
  struct S {
    int data;  // expected-note 2{{'data' declared here}}
    void m_fn1() {
      data_  // expected-error {{use of undeclared identifier 'data_'}}
          [] =  // expected-error {{expected expression}}
          f(data_);  // expected-error {{use of undeclared identifier 'data_'}}
    }
  };
}

namespace testArraySubscriptIndex {
  struct S {
    int data;  // expected-note {{'data' declared here}}
    void m_fn1() {
      (+)[data_];  // expected-error{{expected expression}} expected-error {{use of undeclared identifier 'data_'; did you mean 'data'}}
    }
  };
}

namespace crash_has_include {
int has_include(int); // expected-note {{'has_include' declared here}}
// expected-error@+1 {{'__has_include' must be used within a preprocessing directive}}
int foo = __has_include(42); // expected-error {{use of undeclared identifier '__has_include'; did you mean 'has_include'?}}
}

namespace PR24781_using_crash {
namespace A {
namespace B {
class Foofoo {};  // expected-note {{'A::B::Foofoo' declared here}}
}
}

namespace C {
namespace D {
class Bar : public A::B::Foofoo {};
}
}

using C::D::Foofoo;  // expected-error {{no member named 'Foofoo' in namespace 'PR24781_using_crash::C::D'; did you mean 'A::B::Foofoo'?}}
}

int d = ? L : d; // expected-error {{expected expression}} expected-error {{undeclared identifier}}

struct B0 {
  int : 0 |         // expected-error {{invalid operands to binary expression}}
      (struct B0)e; // expected-error {{use of undeclared identifier}}
};

namespace {
struct a0is0 {};
struct b0is0 {};
int g() {
  0 [
      sizeof(c0is0)]; // expected-error {{use of undeclared identifier}}
};
}

namespace avoidRedundantRedefinitionErrors {
class Class {
  void function(int pid); // expected-note {{'function' declared here}}
};

void Class::function2(int pid) { // expected-error {{out-of-line definition of 'function2' does not match any declaration in 'avoidRedundantRedefinitionErrors::Class'; did you mean 'function'?}}
}

// Expected no redefinition error here.
void Class::function(int pid) { // expected-note {{previous definition is here}}
}

void Class::function(int pid) { // expected-error {{redefinition of 'function'}}
}

namespace ns {
void create_test(); // expected-note {{'create_test' declared here}}
}

void ns::create_test2() { // expected-error {{out-of-line definition of 'create_test2' does not match any declaration in namespace 'avoidRedundantRedefinitionErrors::ns'; did you mean 'create_test'?}}
}

// Expected no redefinition error here.
void ns::create_test() {
}
}
