// RUN: %clang_cc1 -fsyntax-only -std=c23 -pedantic -Wall -Wno-comment -verify=both,c23 %s
// RUN: %clang_cc1 -fsyntax-only -std=c17 -pedantic -Wall -Wno-comment -Wno-c23-extensions -verify=both,c17 %s

/* WG14 N3037:
 * Improved tag compatibility
 *
 * Identical tag types have always been compatible across TU boundaries. This
 * paper made identical tag types compatible within the same TU.
 */

struct foo { int a; } p;

void baz(struct foo f); // c17-note {{passing argument to parameter 'f' here}}

void bar(void) {
  struct foo { int a; } q = {};
  baz(q); // c17-error {{passing 'struct foo' to parameter of incompatible type 'struct foo'}}
}

#define PRODUCT(A ,B) struct prod { A a; B b; }                   // expected-note 2 {{expanded from macro 'PRODUCT'}}
#define SUM(A, B) struct sum { _Bool flag; union { A a; B b; }; } // expected-note 2 {{expanded from macro 'SUM'}}

void func1(PRODUCT(int, SUM(float, double)) x); // both-warning {{declaration of 'struct prod' will not be visible outside of this function}} \
                                                   both-warning {{declaration of 'struct sum' will not be visible outside of this function}} \
                                                   c17-note {{passing argument to parameter 'x' here}}
void func2(PRODUCT(int, SUM(float, double)) y) { // both-warning {{declaration of 'struct prod' will not be visible outside of this function}} \
                                                    both-warning {{declaration of 'struct sum' will not be visible outside of this function}}
  func1(y); // c17-error {{passing 'struct prod' to parameter of incompatible type 'struct prod'}}
}

struct foop { struct { int x; }; }; // c17-note {{previous definition is here}}
struct foop { struct { int x; }; }; // c17-error {{redefinition of 'foop'}}
union barp { int x; float y; };     // c17-note {{previous definition is here}}
union barp { int x; float y; };     // c17-error {{redefinition of 'barp'}}
typedef struct q { int x; } q_t;    // c17-note 2 {{previous definition is here}}
typedef struct q { int x; } q_t;    // c17-error {{redefinition of 'q'}} \
                                       c17-error-re {{typedef redefinition with different types ('struct (unnamed struct at {{.*}})' vs 'struct q')}}
void func3(void) {
  struct S { int x; };       // c17-note {{previous definition is here}}
  struct T { struct S s; };  // c17-note {{previous definition is here}}
  struct S { int x; };       // c17-error {{redefinition of 'S'}}
  struct T { struct S s; };  // c17-error {{redefinition of 'T'}}
}

struct food { int (*p)[3]; }; // c23-note {{field 'p' has type 'int (*)[3]' here}} \
                                 c17-note {{previous definition is here}}
struct food { int (*p)[]; };  // c23-error {{type 'struct food' has incompatible definitions}} \
                                 c23-note {{field 'p' has type 'int (*)[]' here}} \
                                 c17-error {{redefinition of 'food'}}
union bard { int x; float y; }; // c23-note {{field has name 'x' here}} \
                                   c17-note {{previous definition is here}}
union bard { int z; float y; }; // c23-error {{type 'union bard' has incompatible definitions}} \
                                   c23-note {{field has name 'z' here}} \
                                   c17-error {{redefinition of 'bard'}}
union purr { int x; float y; }; // c23-note {{field has name 'x' here}} \
                                   c17-note {{previous definition is here}}
union purr { float y; int x; }; // c23-error {{type 'union purr' has incompatible definitions}} \
                                   c23-note {{field has name 'y' here}} \
                                   c17-error {{redefinition of 'purr'}}

// Different attributes on the tag type are an error.
struct [[gnu::packed]] attr_test { // c17-note {{previous definition is here}} \
                                      c23-note {{attribute 'packed' here}}
  int x;
};

struct attr_test { // c17-error {{redefinition of 'attr_test'}} \
                      c23-error {{type 'struct attr_test' has incompatible definitions}} \
                      c23-note {{no corresponding attribute here}}
  int x;
};

struct attr_test_2 { // c17-note {{previous definition is here}} \
                        c23-note {{no corresponding attribute here}}
  int x;
};

struct [[gnu::packed]] attr_test_2 { // c17-error {{redefinition of 'attr_test_2'}} \
                                        c23-error {{type 'struct attr_test_2' has incompatible definitions}} \
                                        c23-note {{attribute 'packed' here}}
  int x;
};

struct [[deprecated]] attr_test_3 { // c17-note {{previous definition is here}} \
                                       c23-note {{attribute 'deprecated' here}}
  int x;
};

struct [[gnu::packed]] attr_test_3 { // c17-error {{redefinition of 'attr_test_3'}} \
                                        c23-error {{type 'struct attr_test_3' has incompatible definitions}} \
                                        c23-note {{attribute 'packed' here}}
  int x;
};

// Same attribute works fine!
struct [[deprecated]] attr_test_4 { // c17-note {{previous definition is here}}
  int x;
};

struct [[deprecated]] attr_test_4 { // c17-error {{redefinition of 'attr_test_4'}}
  int x;
};

// Same attribute with the same arguments is also fine.
struct [[deprecated("testing")]] attr_test_5 { // c17-note {{previous definition is here}}
  int x;
};

struct [[deprecated("testing")]] attr_test_5 { // c17-error {{redefinition of 'attr_test_5'}}
  int x;
};

// Same attribute with different arguments is not allowed. FIXME: it would be
// nicer to explain that the arguments are different, but attribute argument
// handling is already challenging.
struct [[deprecated("testing")]] attr_test_6 { // c17-note {{previous definition is here}} \
                                                  c23-note {{attribute 'deprecated' here}}
  int x;
};

struct [[deprecated("oh no!")]] attr_test_6 { // c17-error {{redefinition of 'attr_test_6'}} \
                                                 c23-error {{type 'struct attr_test_6' has incompatible definitions}} \
                                                 c23-note {{attribute 'deprecated' here}}
  int x;
};

// Different attributes on the fields make them incompatible.
struct field_attr_test_1 { // c17-note {{previous definition is here}}
  int x;
  [[gnu::packed]] int y; // c23-note {{attribute 'packed' here}}
};

struct field_attr_test_1 { // c17-error {{redefinition of 'field_attr_test_1'}} \
                              c23-error {{type 'struct field_attr_test_1' has incompatible definitions}}
  int x;
  int y; // c23-note {{no corresponding attribute here}}
};

struct field_attr_test_2 { // c17-note {{previous definition is here}}
  [[gnu::packed]] int x; // c23-note {{attribute 'packed' here}}
  int y;
};

struct field_attr_test_2 { // c17-error {{redefinition of 'field_attr_test_2'}} \
                              c23-error {{type 'struct field_attr_test_2' has incompatible definitions}}
  int x; // c23-note {{no corresponding attribute here}}
  int y;
};

struct field_attr_test_3 { // c17-note {{previous definition is here}}
  [[gnu::packed]] int x;
  int y;
};

struct field_attr_test_3 { // c17-error {{redefinition of 'field_attr_test_3'}}
  int x [[gnu::packed]];
  int y;
};

struct field_attr_test_4 { // c17-note {{previous definition is here}}
  [[gnu::packed]] int x; // c23-note {{attribute 'packed' here}}
  int y;
};

struct field_attr_test_4 { // c17-error {{redefinition of 'field_attr_test_4'}} \
                              c23-error {{type 'struct field_attr_test_4' has incompatible definitions}}
  [[deprecated]] int x; // c23-note {{attribute 'deprecated' here}}
  int y;
};

struct field_attr_test_5 { // c17-note {{previous definition is here}}
  [[deprecated("testing")]] int x;
  int y;
};

struct field_attr_test_5 { // c17-error {{redefinition of 'field_attr_test_5'}}
  [[deprecated("testing")]] int x;
  int y;
};

// Show that attribute order does not matter.
struct [[deprecated("testing"), gnu::packed]] field_attr_test_6 { // c17-note {{previous definition is here}}
  int x;
  int y;
};

struct [[gnu::packed, deprecated("testing")]] field_attr_test_6 { // c17-error {{redefinition of 'field_attr_test_6'}}
  int x;
  int y;
};

// Show that attribute syntax does matter.
// FIXME: more clearly identify the syntax as the problem.
struct [[gnu::packed]] field_attr_test_7 { // c17-note {{previous definition is here}} \
                                              c23-note {{attribute 'packed' here}}
  int x;
  int y;
};

struct __attribute__((packed)) field_attr_test_7 { // c17-error {{redefinition of 'field_attr_test_7'}} \
                                                      c23-error {{type 'struct field_attr_test_7' has incompatible definitions}} \
                                                      c23-note {{attribute 'packed' here}}
  int x;
  int y;
};

// Show that attribute *spelling* matters. These two attributes share the same
// implementation in the AST, but the spelling still matters.
struct [[nodiscard]] field_attr_test_8 { // c17-note {{previous definition is here}} \
                                            c23-note {{attribute 'nodiscard' here}}
  int x;
  int y;
};

struct [[gnu::warn_unused_result]] field_attr_test_8 { // c17-error {{redefinition of 'field_attr_test_8'}} \
                                                          c23-error {{type 'struct field_attr_test_8' has incompatible definitions}} \
                                                          c23-note {{attribute 'warn_unused_result' here}}
  int x;
  int y;
};


// Show that equivalent field types are not an issue.
typedef int typedef_of_type_int;
struct equivalent_field_types { // c17-note {{previous definition is here}}
  int x;
};

struct equivalent_field_types { // c17-error {{redefinition of 'equivalent_field_types'}}
  typedef_of_type_int x;
};

struct quals_matter { // c17-note {{previous definition is here}}
  int x;              // c23-note {{field 'x' has type 'int' here}}
};

struct quals_matter { // c17-error {{redefinition of 'quals_matter'}} \
                         c23-error {{type 'struct quals_matter' has incompatible definitions}}                         
  const int x;        // c23-note {{field 'x' has type 'const int' here}}
};

struct qual_order_does_not_matter { // c17-note {{previous definition is here}}
  const volatile int x;
};

struct qual_order_does_not_matter { // c17-error {{redefinition of 'qual_order_does_not_matter'}}
  volatile const int x;
};

struct nested { // both-note {{previous definition is here}}
  int x;
  struct nested { // both-error {{nested redefinition of 'nested'}}
    int x;
  };
};

// Show that bit-field order does matter, including anonymous bit-fields.
struct bit_field_1 { // c17-note 2 {{previous definition is here}}
  int a : 1;
  int : 0;           // c23-note {{field has name '' here}}
  int b : 1;
};

struct bit_field_1 { // c17-error {{redefinition of 'bit_field_1'}}
  int a : 1;
  int : 0;
  int b : 1;
};

struct bit_field_1 { // c17-error {{redefinition of 'bit_field_1'}} \
                        c23-error {{type 'struct bit_field_1' has incompatible definitions}}
  int a : 1;
  int b : 1;	     // c23-note {{field has name 'b' here}}
};

struct bit_field_2 { // c17-note {{previous definition is here}}
  int a : 1;
  int b : 1;         // c23-note {{bit-field 'b' has bit-width 1 here}}
};

struct bit_field_2 { // c17-error {{redefinition of 'bit_field_2'}} \
                        c23-error {{type 'struct bit_field_2' has incompatible definitions}}
  int a : 1;
  int b : 2;         // c23-note {{bit-field 'b' has bit-width 2 here}}
};

// Test a bit-field with an attribute.
struct bit_field_3 { // c17-note {{previous definition is here}}
  int a : 1;
  int b : 1; // c23-note {{no corresponding attribute here}}
};

struct bit_field_3 { // c17-error {{redefinition of 'bit_field_3'}} \
                        c23-error {{type 'struct bit_field_3' has incompatible definitions}}
  int a : 1;
  [[deprecated]] int b : 1; // c23-note {{attribute 'deprecated' here}}
};

struct bit_field_4 { // c17-note {{previous definition is here}}
  int a : 1;
  int b : 1;         // c23-note {{bit-field 'b' has bit-width 1 here}}
};

struct bit_field_4 { // c17-error {{redefinition of 'bit_field_4'}} \
                        c23-error {{type 'struct bit_field_4' has incompatible definitions}}
  int a : 1;
  int b;             // c23-note {{field 'b' is not a bit-field}}
};

struct bit_field_5 { // c17-note {{previous definition is here}}
  int a : 1;
  int b;             // c23-note {{field 'b' is not a bit-field}}
};

struct bit_field_5 { // c17-error {{redefinition of 'bit_field_5'}} \
                        c23-error {{type 'struct bit_field_5' has incompatible definitions}}
  int a : 1;
  int b : 1;         // c23-note {{bit-field 'b' has bit-width 1 here}}
};

enum E { A }; // c17-note 2 {{previous definition is here}}
enum E { A }; // c17-error {{redefinition of 'E'}} \
                 c17-error {{redefinition of enumerator 'A'}}

enum Q { D = 1 }; // c17-note 2 {{previous definition is here}}
enum Q { D = D }; // c17-error {{redefinition of 'Q'}} \
                     c17-error {{redefinition of enumerator 'D'}}

// The order of the enumeration constants does not matter, only the values do.
enum X { B = 1, C = 1 + 1 }; // c17-note 3 {{previous definition is here}}
enum X { C = 2, B = 1 };     // c17-error {{redefinition of 'X'}} \
                                c17-error {{redefinition of enumerator 'C'}} \
                                c17-error {{redefinition of enumerator 'B'}}

// Different enumeration constants.
enum Y { YA = 1, YB = 2 }; // c23-note {{enumerator 'YB' with value 2 here}} \
                              c17-note 3 {{previous definition is here}}
enum Y { YA = 1, YB = 3 }; // c23-error {{type 'enum Y' has incompatible definitions}} \
                              c23-note {{enumerator 'YB' with value 3 here}} \
                              c17-error {{redefinition of 'Y'}} \
							  c17-error {{redefinition of enumerator 'YA'}} \
                              c17-error {{redefinition of enumerator 'YB'}}

// Different enumeration names, same named constants.
enum Z1 { ZC = 1 }; // both-note {{previous definition is here}}
enum Z2 { ZC = 1 }; // both-error {{redefinition of enumerator 'ZC'}}

// Test attributes on the enumeration and enumerators.
enum [[deprecated]] enum_attr_test_1 { // c17-note {{previous definition is here}}
  EAT1 [[deprecated]] // c17-note {{previous definition is here}}
};

enum [[deprecated]] enum_attr_test_1 { // c17-error {{redefinition of 'enum_attr_test_1'}}
  EAT1 [[deprecated]] // c17-error {{redefinition of enumerator 'EAT1'}}
};

enum [[deprecated]] enum_attr_test_2 { // c17-note {{previous definition is here}} \
                                          c23-note {{attribute 'deprecated' here}}
  EAT2 // c17-note {{previous definition is here}}
};

enum enum_attr_test_2 { // c17-error {{redefinition of 'enum_attr_test_2'}} \
                           c23-error {{type 'enum enum_attr_test_2' has incompatible definition}} \
                           c23-note {{no corresponding attribute here}}
  EAT2 // c17-error {{redefinition of enumerator 'EAT2'}}
};

enum enum_attr_test_3 { // c17-note {{previous definition is here}} \
                           c23-note {{no corresponding attribute here}}
  EAT3 // c17-note {{previous definition is here}}
};

enum [[deprecated]] enum_attr_test_3 { // c17-error {{redefinition of 'enum_attr_test_3'}} \
                                          c23-error {{type 'enum enum_attr_test_3' has incompatible definition}} \
                                          c23-note {{attribute 'deprecated' here}}
  EAT3 // c17-error {{redefinition of enumerator 'EAT3'}}
};

enum [[clang::flag_enum]] enum_attr_test_4 { // c17-note {{previous definition is here}} \
                                                c23-note {{attribute 'flag_enum' here}}
  EAT4 // c17-note {{previous definition is here}}
};

enum [[deprecated]] enum_attr_test_4 { // c17-error {{redefinition of 'enum_attr_test_4'}} \
                                          c23-error {{type 'enum enum_attr_test_4' has incompatible definition}} \
                                          c23-note {{attribute 'deprecated' here}}
  EAT4 // c17-error {{redefinition of enumerator 'EAT4'}}
};

enum enum_attr_test_5 { // c17-note {{previous definition is here}}
  EAT5 [[deprecated]] // c17-note {{previous definition is here}} \
                         c23-note {{attribute 'deprecated' here}}
};

enum enum_attr_test_5 { // c17-error {{redefinition of 'enum_attr_test_5'}} \
                           c23-error {{type 'enum enum_attr_test_5' has incompatible definition}}
  EAT5 // c17-error {{redefinition of enumerator 'EAT5'}} \
          c23-note {{no corresponding attribute here}}
};

enum enum_attr_test_6 { // c17-note {{previous definition is here}}
  EAT6 // c17-note {{previous definition is here}} \
          c23-note {{no corresponding attribute here}}
};

enum enum_attr_test_6 { // c17-error {{redefinition of 'enum_attr_test_6'}} \
                           c23-error {{type 'enum enum_attr_test_6' has incompatible definition}}
  EAT6 [[deprecated]] // c17-error {{redefinition of enumerator 'EAT6'}} \
                         c23-note {{attribute 'deprecated' here}}
};

// You cannot declare one with a fixed underlying type and the other without a
// fixed underlying type, or a different underlying type. However, it's worth
// showing that the underlying type doesn't change the redefinition behavior.
enum fixed_test_1 : int { FT1 }; // c17-note 2 {{previous definition is here}}
enum fixed_test_1 : int { FT1 }; // c17-error {{redefinition of 'fixed_test_1'}} \
                                    c17-error {{redefinition of enumerator 'FT1'}}

enum fixed_test_2 : int { FT2 };                 // c17-note 2 {{previous definition is here}}
enum fixed_test_2 : typedef_of_type_int { FT2 }; // c17-error {{redefinition of 'fixed_test_2'}} \
                                                    c17-error {{redefinition of enumerator 'FT2'}}

// Test more bizarre situations in terms of where the type is declared. This
// has always been allowed.
struct declared_funny_1 { int x; }
declared_funny_func(struct declared_funny_1 { int x; } arg) { // both-warning {{declaration of 'struct declared_funny_1' will not be visible outside of this function}}
  return declared_funny_func((__typeof__(arg)){ 0 });
}

// However, this is new.
struct Outer {
  struct Inner { // c17-note {{previous definition is here}}
    int x;
  } i;

  enum InnerEnum { // c17-note {{previous definition is here}}
    IE1            // c17-note {{previous definition is here}}
  } j;
};

struct Inner {   // c17-error {{redefinition of 'Inner'}}
  int x;
};

enum InnerEnum { // c17-error {{redefinition of 'InnerEnum'}}
  IE1            // c17-error {{redefinition of enumerator 'IE1'}}
};

void hidden(void) {
  struct hidden_struct { int x; };
}

struct hidden_struct { // This is fine because the previous declaration is not visible.
  int y;
  int z;
};
