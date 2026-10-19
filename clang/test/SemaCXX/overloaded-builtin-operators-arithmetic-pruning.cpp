// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

// The built-in operator candidates that only have arithmetic or vector
// parameter types are only added if every operand might convert to such a
// type. Check that this doesn't lose candidates.

enum class Scoped { A };
enum Unscoped { UA };
typedef int Vec __attribute__((vector_size(16)));

struct ConvTemplate { template <class T> operator T() const; };
struct ConvInt { operator int() const; };
struct ConvConvInt { operator ConvInt() const; };
struct ConvIntRef { operator int &() const; };
struct ConvScoped { operator Scoped() const; };
struct ConvUnscoped { operator Unscoped() const; };
struct ConvPtr { operator int *() const; };
struct ConvVec { operator Vec() const; };
struct ExplicitBool { explicit operator bool() const; };

Scoped operator|(Scoped, Scoped);

void test(Scoped S, Unscoped U, ConvTemplate CT, ConvInt CI, ConvIntRef CIR,
          ConvConvInt CCI, ConvScoped CS, ConvUnscoped CU, ConvPtr CP, ConvVec CV,
          ExplicitBool EB, int I, int *P, Vec V, _Atomic(int) AI) {
  // Conversion function templates might convert to anything.
  (void)(CT == 1); // expected-error {{use of overloaded operator '==' is ambiguous (with operand types 'ConvTemplate' and 'int')}} \
                   // expected-note-re 1+ {{built-in candidate operator==({{.*}}, int)}}
  (void)(1 - CT);  // expected-error {{use of overloaded operator '-' is ambiguous (with operand types 'int' and 'ConvTemplate')}} \
                   // expected-note-re 1+ {{built-in candidate operator-(int, {{.*}})}}

  (void)(CI == 1);
  (void)(CI + CI);
  (void)(CI << CU);
  (void)(-CI);
  (void)(~CU);
  (void)(U + CI);
  (void)(U | U);
  (void)(CI ? U : CI);
  CIR += CI;
  CIR |= CU;
  ++CIR;
  I += CI;
  I <<= CU;
  AI += CI;
  AI |= CI;
  (void)(AI == CI);
  (void)(CP == P);
  (void)(CP + CI);
  (void)(CV + CV);
  (void)(CV == V);
  V += CV;

  (void)(S == Scoped::A);
  (void)(S | S);
  (void)(CS == Scoped::A);

  (void)(S == 1);  // expected-error {{invalid operands to binary expression ('Scoped' and 'int')}} \
                   // expected-note {{no implicit conversion for scoped enum}}
  (void)(S + CI);  // expected-error {{invalid operands to binary expression ('Scoped' and 'ConvInt')}}
  (void)(CS == 1); // expected-error {{invalid operands to binary expression ('ConvScoped' and 'int')}}
  (void)(CCI == 1); // expected-error {{invalid operands to binary expression ('ConvConvInt' and 'int')}}
  (void)(CP == 1); // expected-error {{invalid operands to binary expression ('ConvPtr' and 'int')}}
  (void)(EB == 1); // expected-error {{invalid operands to binary expression ('ExplicitBool' and 'int')}}
  (void)(AI + S);  // expected-error {{invalid operands to binary expression ('_Atomic(int)' and 'Scoped')}} \
                   // expected-note {{no implicit conversion for scoped enum}}
  I += S;          // expected-error {{invalid operands to binary expression ('int' and 'Scoped')}} \
                   // expected-note {{no implicit conversion for scoped enum}}
  (void)(CV + S);  // expected-error {{invalid operands to binary expression ('ConvVec' and 'Scoped')}}
  (void)(-S);      // expected-error {{invalid argument type 'Scoped' to unary expression}}
}
