// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
struct Incomplete *SomeIncomplete; // #INCOMPLETE
enum E{} SomeE;
enum class E2{} SomeE2;

struct CorrectConvert {
  operator int();
} Convert;

struct ExplicitConvertOnly {
  explicit operator int() const; // #EXPL_CONV
} Explicit;

struct AmbiguousConvert{
  operator int(); // #AMBIG_INT
  operator short(); // #AMBIG_SHORT
  operator float();
} Ambiguous;

short some_short();
int some_int();
long some_long();

void Test() {
#pragma acc kernels num_gangs(1)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(1)
  while(1);

#pragma acc parallel num_gangs(1)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(some_short(), some_int(), some_long())
  while(1);

#pragma acc parallel num_gangs(some_short(), some_int(), some_long())
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(some_short(), some_int(), some_long(), SomeE)
  while(1);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'parallel' directive expects maximum of 3, 4 were provided}}
#pragma acc parallel num_gangs(some_short(), some_int(), some_long(), SomeE)
  while(1);

  // expected-error@+1{{too many integer expression arguments provided to OpenACC 'num_gangs' clause: 'kernels' directive expects maximum of 1, 2 were provided}}
#pragma acc kernels num_gangs(1, 2)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(1, 2)
  while(1);

#pragma acc parallel num_gangs(1, 2)
  while(1);

  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel num_gangs(Ambiguous)
  while(1);

  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(NC, SomeE)
  while(1);

  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(SomeE, NC)
  while(1);

  // expected-error@+3{{OpenACC integer expression type 'struct ExplicitConvertOnly' requires explicit conversion to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+1{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel num_gangs(Explicit, NC)
  while(1);

  // expected-error@+4{{OpenACC integer expression type 'struct ExplicitConvertOnly' requires explicit conversion to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+2{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(Explicit, NC)
  while(1);

  // expected-error@+6{{OpenACC integer expression type 'struct ExplicitConvertOnly' requires explicit conversion to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+4{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
  // expected-error@+3{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
#pragma acc parallel num_gangs(Explicit, NC, Ambiguous)
  while(1);

  // expected-error@+7{{OpenACC integer expression type 'struct ExplicitConvertOnly' requires explicit conversion to 'int'}}
  // expected-note@#EXPL_CONV{{conversion to integral type 'int'}}
  // expected-error@+5{{OpenACC clause 'num_gangs' requires expression of integer type ('struct NotConvertible' invalid)}}
  // expected-error@+4{{multiple conversions from expression type 'struct AmbiguousConvert' to an integral type}}
  // expected-note@#AMBIG_INT{{conversion to integral type 'int'}}
  // expected-note@#AMBIG_SHORT{{conversion to integral type 'short'}}
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(Explicit, NC, Ambiguous)
  while(1);
  // TODO
}

struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;
  static constexpr AmbiguousConvert ACValue;
  static constexpr ExplicitConvertOnly EXValue;

  operator char();
};

template <typename T>
void TestInst() {
  // expected-error@+2{{no member named 'Invalid' in 'HasInt'}}
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(HasInt::Invalid)
  while(1);

  // expected-error@+2{{no member named 'Invalid' in 'HasInt'}}
  // expected-note@#INST{{in instantiation of function template specialization}}
#pragma acc parallel num_gangs(T::Invalid)
  while(1);

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc parallel num_gangs(1, HasInt::Invalid)
  while(1);

  // expected-error@+1{{no member named 'Invalid' in 'HasInt'}}
#pragma acc parallel num_gangs(T::Invalid, 1)
  while(1);

  // expected-error@+2{{no member named 'Invalid' in 'HasInt'}}
  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(1, HasInt::Invalid)
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(T::Invalid, 1)
  while(1);

#pragma acc parallel num_gangs(T::value, typename T::IntTy{})
  while(1);

  // expected-error@+1{{OpenACC 'num_gangs' clause is not valid on 'serial' directive}}
#pragma acc serial num_gangs(T::value, typename T::IntTy{})
  while(1);
}

void Inst() {
  TestInst<HasInt>(); // #INST
}
