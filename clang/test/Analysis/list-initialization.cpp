// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17

template <typename T>
void clang_analyzer_dump(T x);
void clang_analyzer_eval(bool);

template <class FirstT, class... Rest>
void escape(FirstT first, Rest... args);

#include "Inputs/system-header-simulator-cxx.h"

// in C++14 and below, class types with bases are not aggregates, but they
// are in C++17 and above
#if __cplusplus >= 201703L
namespace CXX17_base_class_aggregates {
struct A {
  A();
};

struct B: public A {
};

struct C: public B {
};

struct D: public virtual A {
};

// In C++17, classes B and C are aggregates, so they will be constructed
// without actually calling their trivial constructor. Used to crash.
void foo() {
  B b = {}; // no-crash
  const B &bl = {}; // no-crash
  B &&br = {}; // no-crash

  C c = {}; // no-crash
  const C &cl = {}; // no-crash
  C &&cr = {}; // no-crash

  D d = {}; // no-crash


  C cd = {{}}; // no-crash
  const C &cdl = {{}}; // no-crash
  C &&cdr = {{}}; // no-crash

  const B &bll = {{}}; // no-crash
  const B &bcl = C({{}}); // no-crash
  B &&bcr = C({{}}); // no-crash
}

struct S1 {
  int one;
  int two;
};
struct S2 {
  int three;
};
struct T : public S1, public S2 {
  int four;
  int five;
};
void fields_init_in_order() {
  T direct{1,2,3,4};
  clang_analyzer_eval(direct.one == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(direct.two == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(direct.three == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(direct.four == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(direct.five == 0); // expected-warning{{TRUE}}

  T copy = {1,2,3,4};
  clang_analyzer_eval(copy.one == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.two == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.three == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.four == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.five == 0); // expected-warning{{TRUE}}

  T *ptr = new T{1,2,3,4};
  clang_analyzer_eval(ptr->one == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr->two == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr->three == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr->four == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr->five == 0); // expected-warning{{TRUE}}
  delete ptr;

  T slot;
  T *place = new (&slot) T{1,2,3,4};
  clang_analyzer_eval(place->one == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(place->two == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(place->three == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(place->four == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(place->five == 0); // expected-warning{{TRUE}}
}

struct DefaultOne {
  int one = 1;
};
struct DefaultTwo {
  int two = 2;
};
struct Deriver : public DefaultOne, public DefaultTwo {
};
void empty_lists_bases() {
  Deriver direct{{}, {}};
  Deriver copy = {{}, {}};
  Deriver *ptr = new Deriver{{},{}};

  Deriver slot;
  Deriver *place = new (&slot) Deriver{{},{}};

  clang_analyzer_eval(1 == direct.one); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == direct.two); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.one); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == copy.two); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->one); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == ptr->two); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->one); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == place->two); // expected-warning{{TRUE}}

  delete ptr;
}
} // namespace CXX17_base_class_aggregates

// C++14 and below don't allow list initialization of enums, C++17 and above do
namespace CXX17_enum_list_init {

enum class E {};
const E glob[] = {{}};
void initlistWithinInitlist() {
  // no-crash
  clang_analyzer_dump(glob[0]); // expected-warning-re {{reg_${{[0-9]+}}<enum CXX17_enum_list_init::E Element{glob,0 S64b,enum CXX17_enum_list_init::E}>}}
}

enum UnderlyingStorage : int {
};
void init_value() {
  // No copy list initialization, only direct list init of enum with underlying type
  // is allowed
  UnderlyingStorage direct{1};
  UnderlyingStorage *ptr = new UnderlyingStorage{1};
  UnderlyingStorage slot;
  UnderlyingStorage *place = new (&slot) UnderlyingStorage{1};
  clang_analyzer_eval(1 == direct); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == *ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == *place); // expected-warning{{TRUE}}
}
void empty_list_init() {
  UnderlyingStorage direct{};
  UnderlyingStorage *ptr = new UnderlyingStorage{};
  UnderlyingStorage slot;
  UnderlyingStorage *place = new (&slot) UnderlyingStorage{};
  clang_analyzer_eval(0 == direct); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == *ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == *place); // expected-warning{{TRUE}}
}
} // namespace CXX17_enum_list_init

namespace CXX17_anonymous_union_aggregate_member {
struct S {
  int x;
  union {
    int a;
    int b;
  };
  int y;
};
void no_designated_clause() {
  S direct{1,2};
  clang_analyzer_eval(1 == direct.x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(2 == direct.y); // expected-warning{{FALSE}}
  // FIXME: should be undefined
  clang_analyzer_eval(direct.a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(direct.b); // expected-warning{{UNKNOWN}}

  S copy = {3,4};
  clang_analyzer_eval(3 == copy.x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(4 == copy.y); // expected-warning{{FALSE}}
  // FIXME: should be undefined
  clang_analyzer_eval(copy.a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(copy.b); // expected-warning{{UNKNOWN}}

  S *ptr = new S{5, 6};
  clang_analyzer_eval(5 == ptr->x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(6 == ptr->y); // expected-warning{{FALSE}}
  // FIXME: should be undefined
  clang_analyzer_eval(ptr->a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(ptr->b); // expected-warning{{UNKNOWN}}

  S slot;
  S *place = new (&slot) S{7,8};
  clang_analyzer_eval(7 == place->x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(8 == place->y); // expected-warning{{FALSE}}
  // FIXME: should be undefined
  clang_analyzer_eval(place->a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(place->b); // expected-warning{{UNKNOWN}}

  delete ptr;
}
void designated_clause() {
  S direct{.a = 14};
  clang_analyzer_eval(0 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == direct.y); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(14 == direct.a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(direct.b); // expected-warning{{UNKNOWN}}

  S copy = {.a = 14};
  clang_analyzer_eval(0 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == copy.y); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(14 == copy.a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(copy.b); // expected-warning{{UNKNOWN}}

  S *ptr = new S{.b = 14};
  clang_analyzer_eval(0 == ptr->x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(0 == ptr->y); // expected-warning{{TRUE}}
  // FIXME: should be undefined
  clang_analyzer_eval(ptr->a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(14 == ptr->b); // expected-warning{{UNKNOWN}}

  S slot;
  S *place = new (&slot) S{.b = 14};
  clang_analyzer_eval(0 == place->x); // expected-warning{{TRUE}}
  // FIXME: should be TRUE
  clang_analyzer_eval(0 == place->y); // expected-warning{{TRUE}}
  // FIXME: should be undefined
  clang_analyzer_eval(place->a); // expected-warning{{UNKNOWN}}
  // FIXME: should be undefined
  clang_analyzer_eval(14 == place->b); // expected-warning{{UNKNOWN}}

  delete ptr;
}
} // namespace CXX17_anonymous_union_aggregate_member

// These next are C++14 and up, but just test under C++17
namespace CXX14_default_member_initializers {
struct DMIPreviousField {
  int a;
  int b[2];
  int c = b[a];
};
void dmi_properly_sequenced() {
  DMIPreviousField direct{0, {14, 1}};
  clang_analyzer_eval(0 == direct.a); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == direct.b[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.b[1]); // expected-warning{{TRUE}}
  // FIXME: should be true
  clang_analyzer_eval(14 == direct.c); // expected-warning{{UNKNOWN}}

  DMIPreviousField copy = {0, {14, 1}};
  clang_analyzer_eval(0 == copy.a); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == copy.b[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.b[1]); // expected-warning{{TRUE}}
  // FIXME: should be true
  clang_analyzer_eval(14 == copy.c); // expected-warning{{UNKNOWN}}

  DMIPreviousField *ptr = new DMIPreviousField{0, {14, 1}};
  clang_analyzer_eval(0 == ptr->a); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == ptr->b[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->b[1]); // expected-warning{{TRUE}}
  // FIXME: should be true
  clang_analyzer_eval(14 == ptr->c); // expected-warning{{UNKNOWN}}
  delete ptr;

  DMIPreviousField slot{0, {14, 1}};
  auto place = new (&slot) DMIPreviousField{0, {14, 1}};
  clang_analyzer_eval(0 == place->a); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == place->b[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->b[1]); // expected-warning{{TRUE}}
  // FIXME: should be true
  clang_analyzer_eval(14 == place->c); // expected-warning{{UNKNOWN}}
}

union UDMI {
  int x;
  int y = 1;
};
void union_empty_init_list_default_member_initializer() {
  UDMI direct{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == direct.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(direct.x); // expected-warning{{UNKNOWN}}

  UDMI copy = {};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == copy.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.x); // expected-warning{{UNKNOWN}}

  auto ptr = new UDMI{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == ptr->y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ptr->x); // expected-warning{{UNKNOWN}}

  UDMI slot;
  auto place = new (&slot) UDMI{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == place->y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(place->x); // expected-warning{{UNKNOWN}}
}

struct ArrElt {
  int a = 3;
};
ArrElt const sarr[2] = {};
void array_list_init_aggregates_default_member_initializers() {
  int i = 1;
  // Should this not cause a split?
  clang_analyzer_eval(3 == sarr[i].a); // expected-warning{{TRUE}} expected-warning{{FALSE}}
}
} // CXX14_default_member_initializers

namespace CXX14_same_or_derived_class {
struct T {
  int x;
};
struct U : public T {
  int y;
};
void same_class() {
  T initializer = { 14 };

  T direct{initializer};
  T copy = {initializer};
  T *ptr = new T{initializer};
  T slot;
  auto place = new (&slot) T{initializer};

  clang_analyzer_eval(14 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == place->x); // expected-warning{{TRUE}}

  delete ptr;
}
void derived_class() {
  U initializer;
  initializer.x = 14;

  T direct{initializer};
  T copy = {initializer};
  T *ptr = new T{initializer};
  T slot;
  auto place = new (&slot) T{initializer};

  clang_analyzer_eval(14 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == place->x); // expected-warning{{TRUE}}

  delete ptr;
}
} // namespace CXX14_same_or_derived_class

namespace CXX14_char_array_single_string_clause {
void wchar_string() {
  wchar_t arr[] = { L"12" };
  clang_analyzer_eval(L'1' == arr[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(L'2' == arr[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(L'\0' == arr[2]); // expected-warning{{TRUE}}
}

void u8_string() {
  char arr[] = { u8"12" };
  clang_analyzer_eval(u8'1' == arr[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(u8'2' == arr[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(u8'\0' == arr[2]); // expected-warning{{TRUE}}
}

void u16_string() {
  char16_t arr[] = { u"12" };
  clang_analyzer_eval(u'1' == arr[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(u'2' == arr[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(u'\0' == arr[2]); // expected-warning{{TRUE}}
}

void u32_string() {
  char32_t arr[] = { U"12" };
  clang_analyzer_eval(U'1' == arr[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(U'2' == arr[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(U'\0' == arr[2]); // expected-warning{{TRUE}}
}
} // namespace CXX14_char_array_single_string_clause

namespace CXX17_designated_clauses {
struct S {
  int foo;
  int bar;
};
void one_designated_one_not() {
  S direct{1, .bar = 13};
  clang_analyzer_eval(1 == direct.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == direct.bar); // expected-warning{{TRUE}}

  S copy = {1, .bar = 13};
  clang_analyzer_eval(1 == copy.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == copy.bar); // expected-warning{{TRUE}}

  S *ptr = new S{1, .bar = 13};
  clang_analyzer_eval(1 == ptr->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == ptr->bar); // expected-warning{{TRUE}}
  delete ptr;

  S slot;
  S *place = new (&slot) S{1, .bar = 13};
  clang_analyzer_eval(1 == place->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == place->bar); // expected-warning{{TRUE}}
}
void all_designated() {
  S direct{ .foo = 13, .bar = 1 };
  clang_analyzer_eval(13 == direct.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.bar); // expected-warning{{TRUE}}

  S copy = { .foo = 13, .bar = 1 };
  clang_analyzer_eval(13 == copy.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.bar); // expected-warning{{TRUE}}

  S *ptr = new S{ .foo = 13, .bar = 1 };
  clang_analyzer_eval(13 == ptr->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->bar); // expected-warning{{TRUE}}
  delete ptr;

  S slot;
  S *place = new (&slot) S{ .foo = 13, .bar = 1 };
  clang_analyzer_eval(13 == place->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->bar); // expected-warning{{TRUE}}
}

class PubClass {
public:
  int foo;
  int bar;
};
void public_class_designated_initializers() {
  PubClass direct{
    .foo = 13,
    .bar = 1,
  };
  clang_analyzer_eval(13 == direct.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.bar); // expected-warning{{TRUE}}

  PubClass copy = {
    .foo = 13,
    .bar = 1,
  };
  clang_analyzer_eval(13 == copy.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.bar); // expected-warning{{TRUE}}

  PubClass *ptr = new PubClass{
      .foo = 13,
      .bar = 1,
  };
  clang_analyzer_eval(13 == ptr->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->bar); // expected-warning{{TRUE}}
  delete ptr;

  PubClass slot;
  PubClass *place = new (&slot) PubClass{
    .foo = 13,
    .bar = 1,
  };
  clang_analyzer_eval(13 == place->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->bar); // expected-warning{{TRUE}}
}
  
struct Three {
  int x;
  int y;
  int z;
};
void designated_initializers_with_gaps() {
  Three direct{
    .x = 13,
    .z = 1,
  };
  clang_analyzer_eval(13 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == direct.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.z); // expected-warning{{TRUE}}

  Three copy = {
    .x = 13,
    .z = 1,
  };
  clang_analyzer_eval(13 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == copy.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.z); // expected-warning{{TRUE}}

  Three *ptr = new Three{
    .x = 13,
    .z = 1,
  };
  clang_analyzer_eval(13 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == ptr->y); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->z); // expected-warning{{TRUE}}
  delete ptr;

  Three slot;
  Three *place = new (&slot) Three{
    .x = 13,
    .z = 1,
  };
  clang_analyzer_eval(13 == place->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == place->y); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->z); // expected-warning{{TRUE}}
}

struct Inner {
  int bar;
};
struct Nested {
  int foo;
  Inner inner;
  int baz;
};
void nested_aggregates() {
  auto N1 = new Nested{.baz = 14};
  clang_analyzer_eval(0 == N1->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N1->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == N1->baz); // expected-warning{{TRUE}}

  auto N2 = new Nested{1, {.bar = 2}, 3};
  clang_analyzer_eval(1 == N2->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N2->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N2->baz); // expected-warning{{TRUE}}

  escape(N1,N2);
}
} // namespace CXX17_designated_clauses
#endif // __cplusplus >= 201703L

// Common across different C++ versions
namespace common {

namespace primitive_type {
void int_list_value_init() {
  int vidirect{};
  clang_analyzer_eval(0 == vidirect); // expected-warning{{TRUE}}

  int vicopy = {};
  clang_analyzer_eval(0 == vicopy); // expected-warning{{TRUE}}

  auto viptr = new int{};
  clang_analyzer_eval(0 == *viptr); // expected-warning{{TRUE}}
  delete viptr;

  int vislot;
  auto viplace = new (&vislot) int{};
  clang_analyzer_eval(0 == *viplace); // expected-warning{{TRUE}}

  int fourteendirect{14};
  clang_analyzer_eval(14 == fourteendirect); // expected-warning{{TRUE}}

  int fourteencopy = {14};
  clang_analyzer_eval(14 == fourteencopy); // expected-warning{{TRUE}}

  auto fourteenptr = new int{14};
  clang_analyzer_eval(14 == *fourteenptr); // expected-warning{{TRUE}}
  delete fourteenptr;

  int fourteenslot;
  auto fourteenplace = new (&fourteenslot) int{14};
  clang_analyzer_eval(14 == *fourteenplace); // expected-warning{{TRUE}}
}
} //namespace primitive_type

namespace non_union_class_type {
struct S {
  int foo;
  int bar;
};
void none_designated() {
  S direct{13,1};
  clang_analyzer_eval(13 == direct.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.bar); // expected-warning{{TRUE}}

  S copy = {13,1};
  clang_analyzer_eval(13 == copy.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.bar); // expected-warning{{TRUE}}

  S *ptr = new S{13,1};
  clang_analyzer_eval(13 == ptr->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->bar); // expected-warning{{TRUE}}
  delete ptr;

  S slot;
  S *place = new (&slot) S{13,1};
  clang_analyzer_eval(13 == place->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->bar); // expected-warning{{TRUE}}
}
void none_designated_swapped() {
  S direct{1,13};
  clang_analyzer_eval(1 == direct.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == direct.bar); // expected-warning{{TRUE}}

  S copy = {1,13};
  clang_analyzer_eval(1 == copy.foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == copy.bar); // expected-warning{{TRUE}}

  S *ptr = new S{1,13};
  clang_analyzer_eval(1 == ptr->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == ptr->bar); // expected-warning{{TRUE}}
  delete ptr;

  S slot;
  S *place = new (&slot) S{1,13};
  clang_analyzer_eval(1 == place->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == place->bar); // expected-warning{{TRUE}}
}

class DefaultCtor {
public:
  int x;
  DefaultCtor() : x(1) {}
};
void default_ctor_empty_list_init() {
  DefaultCtor direct{};
  DefaultCtor copy = {};
  DefaultCtor *ptr = new DefaultCtor{};
  DefaultCtor slot;
  auto place = new (&slot) DefaultCtor{};

  clang_analyzer_eval(1 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->x); // expected-warning{{TRUE}}

  delete ptr;
}
void const_lvalue_ref_list_init() {
  const DefaultCtor &direct{};
  const DefaultCtor &copy = {};
  clang_analyzer_eval(1 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.x); // expected-warning{{TRUE}}
}

class NonAggregateImplicitDefaultCtor {
  int NeverUsed = 14;
public:
  int x;
};
void implicit_default_ctor_value_initialized() {
  NonAggregateImplicitDefaultCtor direct{};
  NonAggregateImplicitDefaultCtor copy = {};
  NonAggregateImplicitDefaultCtor *ptr = new NonAggregateImplicitDefaultCtor{};
  NonAggregateImplicitDefaultCtor slot;
  auto place = new (&slot) NonAggregateImplicitDefaultCtor{};
  clang_analyzer_eval(0 == direct.x); // expected-warning{{TRUE}};
  clang_analyzer_eval(0 == copy.x); // expected-warning{{TRUE}};
  clang_analyzer_eval(0 == ptr->x); // expected-warning{{TRUE}};
  clang_analyzer_eval(0 == place->x); // expected-warning{{TRUE}};
  delete ptr;
}

struct NonAggrType {
  int NeverUsed;
  NonAggrType() {}
};
struct AggrWithNonAggr {
  int x;
  NonAggrType y;
};
void leftover_nonaggr_value_init() {
  AggrWithNonAggr direct{1};
  // FIXME: should be TRUE, no error
  clang_analyzer_eval(0 == direct.y.NeverUsed); // expected-warning{{The right operand of '==' is a garbage value}}

  AggrWithNonAggr copy = {1};
  // FIXME: should be TRUE
  clang_analyzer_eval(0 == copy.y.NeverUsed);

  auto ptr = new AggrWithNonAggr{1};
  // FIXME: should be TRUE
  clang_analyzer_eval(0 == ptr->y.NeverUsed);
  delete ptr;

  AggrWithNonAggr slot;
  auto place = new (&slot) AggrWithNonAggr{1};
  // FIXME: should be TRUE
  clang_analyzer_eval(0 == place->y.NeverUsed);
}

struct Empty {
};
struct NonEmpty {
  Empty emp;
  int x;
};
struct NonEmptyLast {
  int x;
  Empty emp;
};
void empty_skipped() {
  NonEmpty direct{{}, 14};
  NonEmpty copy = {{}, 14};
  NonEmpty *ptr = new NonEmpty{{}, 14};
  NonEmpty slot;
  auto place = new (&slot) NonEmpty{{}, 14};
  clang_analyzer_eval(14 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == place->x); // expected-warning{{TRUE}}
  delete ptr;
}
void empty() {
  Empty E{}; // no crash
}
void empty_last() {
  NonEmptyLast direct{1};
  NonEmptyLast copy = {1};
  auto ptr = new NonEmptyLast{1};
  NonEmptyLast slot;
  auto place = new (&slot) NonEmptyLast{1};
  clang_analyzer_eval(1 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->x); // expected-warning{{TRUE}}
}

struct ImplicitConversion {
  int x;
  operator int() { return x; }
};
struct AggregateMembersImplicit {
  ImplicitConversion x1, x2;
  int z;
};
void aggregate_implicit_conversions() {
  ImplicitConversion ToConvert{14};
  AggregateMembersImplicit direct{0,1,ToConvert};
  AggregateMembersImplicit copy = {0,1,ToConvert};
  auto ptr = new AggregateMembersImplicit{0,1,ToConvert};
  AggregateMembersImplicit slot;
  auto place = new (&slot) AggregateMembersImplicit{0,1,ToConvert};

  clang_analyzer_eval(0 == direct.x1.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.x2.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == direct.z); // expected-warning{{TRUE}}

  clang_analyzer_eval(0 == copy.x1.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == copy.x2.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == copy.z); // expected-warning{{TRUE}}

  clang_analyzer_eval(0 == ptr->x1.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == ptr->x2.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == ptr->z); // expected-warning{{TRUE}}

  clang_analyzer_eval(0 == place->x1.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == place->x2.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == place->z); // expected-warning{{TRUE}}

  delete ptr;
}

struct Three {
  int x;
  int y;
  int z;
};
void initializer_clauses_sequenced() {
  // FIXME: should not warn
  int count = 0;
  Three direct{count++, count++, count++};
  Three copy = {count++, count++, count++};
  auto ptr = new Three{count++, count++, count++};
  Three slot;
  auto place = new (&slot) Three{count++, count++, count++};

  clang_analyzer_eval(0 == direct.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == direct.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == direct.z); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == copy.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == copy.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(5 == copy.z); // expected-warning{{TRUE}}
  clang_analyzer_eval(6 == ptr->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(7 == ptr->y); // expected-warning{{TRUE}}
  clang_analyzer_eval(8 == ptr->z); // expected-warning{{TRUE}}
  clang_analyzer_eval(9 == place->x); // expected-warning{{TRUE}}
  clang_analyzer_eval(10 == place->y); // expected-warning{{TRUE}}
  clang_analyzer_eval(11 == place->z); // expected-warning{{TRUE}}

  delete ptr;
}

// https://eel.is/c++draft/dcl.init.aggr#note-6:
// Static data members, non-static data members of anonymous
// union members, and unnamed bit-fields are not considered
// elements of the aggregate.
struct NonConsideredFields {
  int i;
  static int s;
  int j;
  int :17;
  int k;
};
void considered_fields_initd() {
  NonConsideredFields direct{1, 2, 3};
  clang_analyzer_eval(1 == direct.i); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == direct.j); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == direct.k); // expected-warning{{TRUE}}

  NonConsideredFields copy = {1, 2, 3};
  clang_analyzer_eval(1 == copy.i); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == copy.j); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == copy.k); // expected-warning{{TRUE}}

  auto ptr = new NonConsideredFields { 1, 2, 3 };
  clang_analyzer_eval(1 == ptr->i); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == ptr->j); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == ptr->k); // expected-warning{{TRUE}}
  delete ptr;

  NonConsideredFields slot;
  auto place = new (&slot) NonConsideredFields { 1, 2, 3 };
  clang_analyzer_eval(1 == place->i); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == place->j); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == place->k); // expected-warning{{TRUE}}
}

struct Inner {
  int bar;
};
struct Nested {
  int foo;
  Inner inner;
  int baz;
};
void nested_aggregates() {
  auto N = new Nested{};
  clang_analyzer_eval(0 == N->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N->baz); // expected-warning{{TRUE}}

  auto N1 = new Nested{1};
  clang_analyzer_eval(1 == N1->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N1->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N1->baz); // expected-warning{{TRUE}}

  auto N2 = new Nested{.baz = 14};
  clang_analyzer_eval(0 == N2->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N2->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == N2->baz); // expected-warning{{TRUE}}

  auto N3 = new Nested{1,2,3};
  clang_analyzer_eval(1 == N3->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N3->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N3->baz); // expected-warning{{TRUE}}

  auto N4 = new Nested{1, {}, 3};
  clang_analyzer_eval(1 == N4->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N4->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N4->baz); // expected-warning{{TRUE}}

  auto N5 = new Nested{{},{},{}};
  clang_analyzer_eval(0 == N5->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N5->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N5->baz); // expected-warning{{TRUE}}

  auto N6 = new Nested{1, {.bar = 2}, 3};
  clang_analyzer_eval(1 == N6->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N6->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N6->baz); // expected-warning{{TRUE}}

  auto N7 = new Nested{1, {2}, 3};
  clang_analyzer_eval(1 == N7->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N7->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N7->baz); // expected-warning{{TRUE}}

  escape(N,N1,N2,N3,N4,N5,N6,N7);
}
} // namespace non_union_class_type

namespace arrays {
int const glob_arr1[3] = {};
void array_empty_list_init_values() {
  clang_analyzer_eval(glob_arr1[0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[2] == 0); // expected-warning{{TRUE}}
}
void array_empty_list_init_invalid_index_undef() {
  const int *ptr = glob_arr1;
  int idx = -42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}
void array_empty_list_init_symbolic_index_unknown(int idx) {
  clang_analyzer_dump(glob_arr1[idx]); // expected-warning{{Unknown}}
}

int const glob_arr4[4][2] = {};
void array_nested_empty_list_init_values() {
  clang_analyzer_eval(glob_arr4[0][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[0][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[1][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[1][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[2][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[2][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[3][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[3][1] == 0); // expected-warning{{TRUE}}
}
void array_nested_empty_list_init_invalid_idx_undef() {
  int idx = -42;
  auto x = glob_arr4[1][idx]; // expected-warning{{uninitialized}}
}
void array_nested_empty_list_init_invalid_idx_undef2() {
  const int *ptr = glob_arr4[1];
  int idx = -42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}

int const glob_arr2[4] = {1, 2};
void array_fewer_init_clauses_values() {
  int const *ptr = glob_arr2;
  clang_analyzer_eval(ptr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[2] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[3] == 0); // expected-warning{{TRUE}}
}
void array_fewer_init_clauses_values_invalid_index() {
  const int *ptr = glob_arr2;
  int idx = 42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}

int const glob_arr5[4][2] = {{1}, 3, 4, 5};
void array_nested_init_list() {
  clang_analyzer_eval(glob_arr5[0][0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[0][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[1][0] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[1][1] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[2][0] == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[2][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[3][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr5[3][1] == 0); // expected-warning{{TRUE}}
}
void array_nested_init_list_oob_read() {
  int const *ptr = glob_arr5[1];
  clang_analyzer_eval(ptr[0] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[1] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[2] == 5); // expected-warning{{out of bounds}}
}
void array_nested_init_list_invalid_index() {
  int idx = -42;
  auto x = glob_arr5[1][idx]; // expected-warning{{uninitialized}}
}
void array_nested_init_list_invalid_index2() {
  int const *ptr = &glob_arr5[1][0];
  int idx = 42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}

char const char_string_init[5] = {"123"};
void array_char_init_with_char_string() {
  clang_analyzer_eval(char_string_init[0] == '1');  // expected-warning{{TRUE}}
  clang_analyzer_eval(char_string_init[1] == '2');  // expected-warning{{TRUE}}
  clang_analyzer_eval(char_string_init[2] == '3');  // expected-warning{{TRUE}}
  clang_analyzer_eval(char_string_init[3] == '\0'); // expected-warning{{TRUE}}
  clang_analyzer_eval(char_string_init[4] == '\0'); // expected-warning{{TRUE}}
}
void array_char_init_with_char_string_invalid_index() {
  int idx = -42;
  auto x = char_string_init[idx]; // expected-warning{{uninitialized}}
}
void array_char_init_with_char_string_invalid_index2() {
  const char *ptr = char_string_init;
  int idx = 42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}

const int glob_arr9[2][4] = {{(1), 2, ((3)), 4}, 5, 6, (((7)))};
void array_list_init_with_parens() {
  clang_analyzer_eval(glob_arr9[0][0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[0][1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[0][2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[0][3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[1][0] == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[1][1] == 6); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[1][2] == 7); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr9[1][3] == 0); // expected-warning{{TRUE}}
}

void array_list_init_first_column_only() {
  int arr[2][2] = { {1}, {1} };
  clang_analyzer_eval(1 == arr[0][0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == arr[0][1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == arr[1][0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == arr[1][1]); // expected-warning{{TRUE}}
}

struct NonAggrValueInitArrays {
  int y;
  int x[2];
  NonAggrValueInitArrays() {}
};
void array_nonaggr_list_value_init() {
  NonAggrValueInitArrays direct{};
  // FIXME: both should be true, no error
  clang_analyzer_eval(0 == direct.x[0]); // expected-warning{{The right operand of '==' is a garbage value}}
  clang_analyzer_eval(0 == direct.x[1]);

  NonAggrValueInitArrays copy = {};
  // FIXME: both should be true
  clang_analyzer_eval(0 == copy.x[0]);
  clang_analyzer_eval(0 == copy.x[1]);

  NonAggrValueInitArrays *ptr = new NonAggrValueInitArrays{};
  // FIXME: both should be true
  clang_analyzer_eval(0 == ptr->x[0]);
  clang_analyzer_eval(0 == ptr->x[1]);
  delete ptr;

  NonAggrValueInitArrays slot;
  auto place = new (&slot) NonAggrValueInitArrays{};
  // FIXME: both should be true
  clang_analyzer_eval(0 == ptr->x[0]);
  clang_analyzer_eval(0 == ptr->x[1]);
}

struct ValueInitArraysLeftOvers {
  int init;
  int x[2];
};
void array_aggr_list_value_init_leftovers() {
  ValueInitArraysLeftOvers direct{1};
  clang_analyzer_eval(0 == direct.x[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == direct.x[1]); // expected-warning{{TRUE}}

  ValueInitArraysLeftOvers copy = {1};
  clang_analyzer_eval(0 == copy.x[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == copy.x[1]); // expected-warning{{TRUE}}

  ValueInitArraysLeftOvers *ptr = new ValueInitArraysLeftOvers{1};
  clang_analyzer_eval(0 == ptr->x[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == ptr->x[1]); // expected-warning{{TRUE}}
  delete ptr;

  ValueInitArraysLeftOvers slot;
  auto place = new (&slot) ValueInitArraysLeftOvers{1};
  clang_analyzer_eval(0 == place->x[0]); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == place->x[1]); // expected-warning{{TRUE}}
}

void array_initializer_clauses_sequenced() {
  // FIXME: should not warn
  int direct_arr[3]{{}, 1 + direct_arr[0], 1 + direct_arr[1]}; // expected-warning{{The right operand of '+' is a garbage value}}
  int copy_arr[3] = {{}, 1 + copy_arr[0], 1 + copy_arr[1]};

  // FIXME: these should be TRUE
  clang_analyzer_eval(0 == direct_arr[0]);
  clang_analyzer_eval(1 == direct_arr[1]);
  clang_analyzer_eval(2 == direct_arr[2]);
  clang_analyzer_eval(0 == copy_arr[0]);
  clang_analyzer_eval(1 == copy_arr[1]);
  clang_analyzer_eval(2 == copy_arr[2]);
}

struct S {
  int foo;
  int bar;
};
void non_designated_array_of_aggr_struct() {
  S direct[2]{ {1, 2}, {3, 4} };
  clang_analyzer_eval(1 == direct[0].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == direct[0].bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == direct[1].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == direct[1].bar); // expected-warning{{TRUE}}

  S copy[2]{ {1, 2}, {3, 4} };
  clang_analyzer_eval(1 == copy[0].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == copy[0].bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == copy[1].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == copy[1].bar); // expected-warning{{TRUE}}

  S *s = new S[2] { {1, 2}, {3, 4} };
  clang_analyzer_eval(1 == s[0].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == s[0].bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == s[1].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == s[1].bar); // expected-warning{{TRUE}}
  delete[] s;

  S slot[2];
  S *place = new (&slot) S[2] { {1, 2}, {3, 4} };
  clang_analyzer_eval(1 == place[0].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == place[0].bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == place[1].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == place[1].bar); // expected-warning{{TRUE}}
}

} // namespace arrays

namespace std_initializer_lists {
struct C {
  C(std::initializer_list<int *> list);
};
void testPointerEscapeIntoLists() {
  C empty{}; // no-crash

  // Do not warn that 'x' leaks. It might have been deleted by
  // the destructor of 'c'.
  int *x = new int;
  C c{x}; // no-warning
}

void testPassListsWithExplicitConstructors() {
  (void)(std::initializer_list<int>){12}; // no-crash
}
} // namespace std_initializer_lists

namespace unions {
union U {
  int x;
  int y;
};
void union_empty_init_list() {
  U direct{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(0 == direct.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(direct.y); // expected-warning{{UNKNOWN}}

  U copy = {};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(0 == copy.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.y); // expected-warning{{UNKNOWN}}

  auto ptr = new U{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(0 == ptr->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ptr->y); // expected-warning{{UNKNOWN}}
  delete ptr;

  U slot;
  auto place = new (&slot) U{};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(0 == place->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(place->y); // expected-warning{{UNKNOWN}}
}
void union_single_init_clause() {
  U direct{1};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == direct.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(direct.y); // expected-warning{{UNKNOWN}}

  U copy = {1};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == copy.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.y); // expected-warning{{UNKNOWN}}

  auto ptr = new U{1};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == ptr->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ptr->y); // expected-warning{{UNKNOWN}}
  delete ptr;

  U slot;
  auto place = new (&slot) U{1};
  // FIXME: should be TRUE and undefined
  clang_analyzer_eval(1 == place->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(place->y); // expected-warning{{UNKNOWN}}
}
void union_single_initializer_clause_first_field() {
  U direct{.x = 1};
  U copy = {.x = 1};
  U *ptr = new U{.x = 1};
  U slot;
  U *place = new (&slot) U{.x = 1};

  // FIXME: should be true
  clang_analyzer_eval(1 == direct.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == copy.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == ptr->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == place->x); // expected-warning{{UNKNOWN}}

  // FIXME: should be undefined
  clang_analyzer_eval(1 + direct.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + copy.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + ptr->y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + place->y); // expected-warning{{UNKNOWN}}

  delete ptr;
}
void union_single_initializer_clause_non_first_field() {
  U direct{.y = 1};
  U copy = {.y = 1};
  U *ptr = new U{.y = 1};
  U slot;
  U *place = new (&slot) U{.y = 1};

  // FIXME: should be true
  clang_analyzer_eval(1 == direct.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == copy.y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == ptr->y); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 == place->y); // expected-warning{{UNKNOWN}}

  // FIXME: should be undefined
  clang_analyzer_eval(1 + direct.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + copy.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + ptr->x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(1 + place->x); // expected-warning{{UNKNOWN}}

  delete ptr;
}

struct Empty {};
union EmptyFirst {
  Empty e;
  int x;
};
void union_empty_list_init_no_crash() {
  EmptyFirst direct{};
  EmptyFirst copy = {};
  EmptyFirst *ptr = new EmptyFirst{};
  EmptyFirst slot;
  auto place = new (&slot) EmptyFirst{};
  delete ptr;
}
void union_empty_list_init_empty_list_init_no_crash() {
  EmptyFirst direct{{}};
  EmptyFirst copy = {{}};
  EmptyFirst *ptr = new EmptyFirst{{}};
  EmptyFirst slot;
  auto place = new (&slot) EmptyFirst{{}};
  delete ptr;
}
} // namespace unions

namespace transparent_init_list_exprs {
class A {};

class B: private A {};

B boo();
void foo1() {
  B b { boo() }; // no-crash
}

class C: virtual public A {};

C coo();
void foo2() {
  C c { coo() }; // no-crash
}

B foo_recursive() {
  B b { foo_recursive() };
  return b;
}
} // namespace transparent_init_list_exprs
} // namespace common
