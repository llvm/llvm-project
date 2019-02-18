// RUN: %clang_cc1 -std=c++11 -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

template <int V, int I>
void test_nontype_template_param(int *List, int Length) {
#pragma clang loop vectorize_width(V) interleave_count(I)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(V + 4) interleave_count(I + 4)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

template <int V>
void test_nontype_template_vectorize(int *List, int Length) {
  /* expected-error {{invalid value '-1'; must be positive}} */ #pragma clang loop vectorize_width(V)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }

  /* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop vectorize_width(V / 2)
  for (int i = 0; i < Length; i++) {
    List[i] += i;
  }
}

template <int I>
void test_nontype_template_interleave(int *List, int Length) {
  /* expected-error {{invalid value '-1'; must be positive}} */ #pragma clang loop interleave_count(I)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }

  /* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop interleave_count(2 % I)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

template <char V>
void test_nontype_template_char(int *List, int Length) {
  /* expected-error {{invalid argument of type 'char'; expected an integer type}} */ #pragma clang loop vectorize_width(V)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

template <bool V>
void test_nontype_template_bool(int *List, int Length) {
  /* expected-error {{invalid argument of type 'bool'; expected an integer type}} */ #pragma clang loop vectorize_width(V)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

template <int V, int I>
void test_nontype_template_badarg(int *List, int Length) {
  /* expected-error {{use of undeclared identifier 'Vec'}} */ #pragma clang loop vectorize_width(Vec) interleave_count(I)
  /* expected-error {{use of undeclared identifier 'Int'}} */ #pragma clang loop vectorize_width(V) interleave_count(Int)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

template <typename T>
void test_type_template_vectorize(int *List, int Length) {
  const T Value = -1;
  /* expected-error {{invalid value '-1'; must be positive}} */ #pragma clang loop vectorize_width(Value)
  for (int i = 0; i < Length; i++) {
    List[i] = i;
  }
}

void test(int *List, int Length) {
  int i = 0;

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
#pragma clang loop unroll(full)
  while (i + 1 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
#pragma clang loop unroll_count(16)
  while (i < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#pragma clang loop unroll(disable)
  while (i - 1 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(4) interleave_count(8) unroll_count(16)
  while (i - 2 < Length) {
    List[i] = i;
  }

#pragma clang loop interleave_count(16)
  while (i - 3 < Length) {
    List[i] = i;
  }

  int VList[Length];
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
  for (int j : VList) {
    VList[j] = List[j];
  }

#pragma clang loop distribute(enable)
  for (int j : VList) {
    VList[j] = List[j];
  }

#pragma clang loop distribute(disable)
  for (int j : VList) {
    VList[j] = List[j];
  }

  test_nontype_template_param<4, 8>(List, Length);

/* expected-error {{expected '('}} */ #pragma clang loop vectorize
/* expected-error {{expected '('}} */ #pragma clang loop interleave
/* expected-error {{expected '('}} */ #pragma clang loop unroll
/* expected-error {{expected '('}} */ #pragma clang loop distribute

/* expected-error {{expected ')'}} */ #pragma clang loop vectorize(enable
/* expected-error {{expected ')'}} */ #pragma clang loop interleave(enable
/* expected-error {{expected ')'}} */ #pragma clang loop unroll(full
/* expected-error {{expected ')'}} */ #pragma clang loop distribute(enable

/* expected-error {{expected ')'}} */ #pragma clang loop vectorize_width(4
/* expected-error {{expected ')'}} */ #pragma clang loop interleave_count(4
/* expected-error {{expected ')'}} */ #pragma clang loop unroll_count(4

/* expected-error {{missing argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop vectorize()
/* expected-error {{missing argument; expected an integer value}} */ #pragma clang loop interleave_count()
/* expected-error {{missing argument; expected 'enable', 'full' or 'disable'}} */ #pragma clang loop unroll()
/* expected-error {{missing argument; expected 'enable' or 'disable'}} */ #pragma clang loop distribute()

/* expected-error {{missing option; expected vectorize, vectorize_width, interleave, interleave_count, unroll, unroll_count, or distribute}} */ #pragma clang loop
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop badkeyword
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop badkeyword(enable)
/* expected-error {{invalid option 'badkeyword'}} */ #pragma clang loop vectorize(enable) badkeyword(4)
/* expected-warning {{extra tokens at end of '#pragma clang loop'}} */ #pragma clang loop vectorize(enable) ,
  while (i-4 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop vectorize_width(0)
/* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop interleave_count(0)
/* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop unroll_count(0)

/* expected-error {{expression is not an integral constant expression}} expected-note {{division by zero}} */ #pragma clang loop vectorize_width(10 / 0)
/* expected-error {{invalid value '0'; must be positive}} */ #pragma clang loop interleave_count(10 / 5 - 2)
  while (i-5 < Length) {
    List[i] = i;
  }

test_nontype_template_vectorize<4>(List, Length);
/* expected-note {{in instantiation of function template specialization}} */ test_nontype_template_vectorize<-1>(List, Length);
test_nontype_template_interleave<8>(List, Length);
/* expected-note {{in instantiation of function template specialization}} */ test_nontype_template_interleave<-1>(List, Length);

/* expected-note {{in instantiation of function template specialization}} */ test_nontype_template_char<'A'>(List, Length); // Loop hint arg cannot be a char.
/* expected-note {{in instantiation of function template specialization}} */ test_nontype_template_bool<true>(List, Length);  // Or a bool.
/* expected-note {{in instantiation of function template specialization}} */ test_type_template_vectorize<int>(List, Length); // Or a template type.

/* expected-error {{value '3000000000' is too large}} */ #pragma clang loop vectorize_width(3000000000)
/* expected-error {{value '3000000000' is too large}} */ #pragma clang loop interleave_count(3000000000)
/* expected-error {{value '3000000000' is too large}} */ #pragma clang loop unroll_count(3000000000)
  while (i-6 < Length) {
    List[i] = i;
  }

/* expected-warning {{extra tokens at end of '#pragma clang loop'}} */ #pragma clang loop vectorize_width(1 +) 1
/* expected-warning {{extra tokens at end of '#pragma clang loop'}} */ #pragma clang loop vectorize_width(1) +1
const int VV = 4;
/* expected-error {{expected expression}} */ #pragma clang loop vectorize_width(VV +/ 2)
/* expected-error {{use of undeclared identifier 'undefined'}} */ #pragma clang loop vectorize_width(VV+undefined)
/* expected-error {{expected ')'}} */ #pragma clang loop vectorize_width(1+(^*/2 * ()
/* expected-warning {{extra tokens at end of '#pragma clang loop' - ignored}} */ #pragma clang loop vectorize_width(1+(-0[0]))))))

/* expected-error {{use of undeclared identifier 'badvalue'}} */ #pragma clang loop vectorize_width(badvalue)
/* expected-error {{use of undeclared identifier 'badvalue'}} */ #pragma clang loop interleave_count(badvalue)
/* expected-error {{use of undeclared identifier 'badvalue'}} */ #pragma clang loop unroll_count(badvalue)
  while (i-6 < Length) {
    List[i] = i;
  }

/* expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop vectorize(badidentifier)
/* expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop interleave(badidentifier)
/* expected-error {{invalid argument; expected 'enable', 'full' or 'disable'}} */ #pragma clang loop unroll(badidentifier)
/* expected-error {{invalid argument; expected 'enable' or 'disable'}} */ #pragma clang loop distribute(badidentifier)
  while (i-7 < Length) {
    List[i] = i;
  }

// PR20069 - Loop pragma arguments that are not identifiers or numeric
// constants crash FE.
/* expected-error {{expected ')'}} */ #pragma clang loop vectorize(()
/* expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}} */ #pragma clang loop interleave(*)
/* expected-error {{invalid argument; expected 'enable', 'full' or 'disable'}} */ #pragma clang loop unroll(=)
/* expected-error {{invalid argument; expected 'enable' or 'disable'}} */ #pragma clang loop distribute(+)
/* expected-error {{type name requires a specifier or qualifier}} expected-error {{expected expression}} */ #pragma clang loop vectorize_width(^)
/* expected-error {{expected expression}} expected-error {{expected expression}} */ #pragma clang loop interleave_count(/)
/* expected-error {{expected expression}} expected-error {{expected expression}} */ #pragma clang loop unroll_count(==)
  while (i-8 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(enable)
/* expected-error {{expected a for, while, or do-while loop to follow '#pragma clang loop'}} */ int j = Length;
  List[0] = List[1];

  while (j-1 < Length) {
    List[j] = j;
  }

// FIXME: A bug in ParsedAttributes causes the order of the attributes to be
// processed in reverse. Consequently, the errors occur on the first of pragma
// of the next three tests rather than the last, and the order of the kinds
// is also reversed.

#pragma clang loop vectorize_width(4)
/* expected-error {{incompatible directives 'vectorize(disable)' and 'vectorize_width(4)'}} */ #pragma clang loop vectorize(disable)
#pragma clang loop interleave_count(4)
/* expected-error {{incompatible directives 'interleave(disable)' and 'interleave_count(4)'}} */ #pragma clang loop interleave(disable)
#pragma clang loop unroll_count(4)
/* expected-error {{incompatible directives 'unroll(disable)' and 'unroll_count(4)'}} */ #pragma clang loop unroll(disable)
  while (i-8 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(enable)
/* expected-error {{duplicate directives 'vectorize(enable)' and 'vectorize(disable)'}} */ #pragma clang loop vectorize(disable)
#pragma clang loop interleave(enable)
/* expected-error {{duplicate directives 'interleave(enable)' and 'interleave(disable)'}} */ #pragma clang loop interleave(disable)
#pragma clang loop unroll(full)
/* expected-error {{duplicate directives 'unroll(full)' and 'unroll(disable)'}} */ #pragma clang loop unroll(disable)
#pragma clang loop distribute(enable)
/* expected-error {{duplicate directives 'distribute(enable)' and 'distribute(disable)'}} */ #pragma clang loop distribute(disable)
  while (i-9 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize(disable)
/* expected-error {{incompatible directives 'vectorize(disable)' and 'vectorize_width(4)'}} */ #pragma clang loop vectorize_width(4)
#pragma clang loop interleave(disable)
/* expected-error {{incompatible directives 'interleave(disable)' and 'interleave_count(4)'}} */ #pragma clang loop interleave_count(4)
#pragma clang loop unroll(disable)
/* expected-error {{incompatible directives 'unroll(disable)' and 'unroll_count(4)'}} */ #pragma clang loop unroll_count(4)
  while (i-10 < Length) {
    List[i] = i;
  }

#pragma clang loop vectorize_width(8)
/* expected-error {{duplicate directives 'vectorize_width(8)' and 'vectorize_width(4)'}} */ #pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
/* expected-error {{duplicate directives 'interleave_count(8)' and 'interleave_count(4)'}} */ #pragma clang loop interleave_count(4)
#pragma clang loop unroll_count(8)
/* expected-error {{duplicate directives 'unroll_count(8)' and 'unroll_count(4)'}} */ #pragma clang loop unroll_count(4)
  while (i-11 < Length) {
    List[i] = i;
  }

#pragma clang loop unroll(full)
/* expected-error {{incompatible directives 'unroll(full)' and 'unroll_count(4)'}} */ #pragma clang loop unroll_count(4)
  while (i-11 < Length) {
    List[i] = i;
  }

#pragma clang loop interleave(enable)
/* expected-error {{expected statement}} */ }
