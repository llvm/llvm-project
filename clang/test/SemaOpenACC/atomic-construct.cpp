// RUN: %clang_cc1 %s -fopenacc -Wno-unused-value -verify 

void NormalFunc(int I) {
  // No clauses are valid, but we parse them anyway, just mark them as not valid
  // on this construct.
 
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'atomic' directive}}
#pragma acc atomic copy(I)
  I = I + 1;
  // expected-error@+1{{OpenACC 'copy' clause is not valid on 'atomic' directive}}
#pragma acc atomic read copy(I)
  I = I;
}

struct Struct{
  Struct *getPtr();
  Struct &operator++();
  Struct &operator--();
  Struct &operator++(int);
  Struct &operator--(int);

  Struct &operator+=(int);
  Struct &operator*=(int);
  Struct &operator-=(int);
  Struct &operator/=(int);
  Struct &operator&=(int);
  Struct &operator|=(int);
  Struct &operator<<=(int);
  Struct &operator>>=(int);
  Struct &operator^=(int);
  Struct &operator%=(int);
  Struct &operator!=(int);
  Struct &operator+();
  Struct &operator-();

  operator int();
  void operator()();
  Struct &operator*();
  Struct &operator=(int);
};

int operator+(Struct&, int);
int operator+(int, Struct&);
Struct &operator+(Struct&, Struct&);
Struct &operator*(Struct&, Struct&);
Struct &operator-(Struct&, Struct&);

Struct S1, S2;

template<typename T>
T &getRValue();

template<typename T>
void AtomicReadTemplate(T LHS, T RHS) {
#pragma acc atomic read
  LHS = RHS;

  T *LHSPtr, *RHSPtr;

#pragma acc atomic read
  LHSPtr = RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{right operand to assignment expression must be an l-value}}
#pragma acc atomic read
  LHS = RHS + 1;

#pragma acc atomic read
  *LHSPtr = RHS;

#pragma acc atomic read
  LHS = *RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{right operand to assignment expression must be an l-value}}
#pragma acc atomic read
  LHS = getRValue<T>();
}

template<typename T>
void AtomicReadTemplate2(T LHS, T RHS) {
  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic read
  LHS = RHS;

  T *LHSPtr, *RHSPtr;
  // Fine, now a pointer.
#pragma acc atomic read
  LHSPtr = RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{right operand to assignment expression must be an l-value}}
#pragma acc atomic read
  LHS = *RHS.getPtr();

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic read
  *LHSPtr = RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic read
  LHS = *RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be an l-value}}
#pragma acc atomic read
  getRValue<T>() = getRValue<T>();

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{right operand to assignment expression must be an l-value}}
#pragma acc atomic read
  LHS = getRValue<T>();
}

void AtomicRead(int LHS, int RHS) {
  AtomicReadTemplate(LHS, RHS);
  AtomicReadTemplate2(S1, S2); // expected-note{{in instantiation of function template specialization}}

#pragma acc atomic read
  LHS = RHS;

  int *LHSPtr, *RHSPtr;

#pragma acc atomic read
  LHSPtr = RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic read
  S1 = S2;

  // expected-error@+2{{statement associated with OpenACC 'atomic read' directive is invalid}}
  // expected-note@+2{{right operand to assignment expression must be an l-value}}
#pragma acc atomic read
  LHS = RHS + 1;

#pragma acc atomic read
  *LHSPtr = RHS;

#pragma acc atomic read
  LHS = *RHSPtr;

  // There is no way to test that = is an overloaded operator, since there
  // really isn't a way to create an operator= without a class type on one side
  // or the other.
}

template<typename T>
void AtomicWriteTemplate(T LHS, T RHS) {
#pragma acc atomic write
  LHS = RHS;

  T *LHSPtr, *RHSPtr;
#pragma acc atomic write
  LHSPtr = RHSPtr;

#pragma acc atomic write
  *LHSPtr = *RHSPtr;

  // allowed, expr is ok.
#pragma acc atomic write
  LHS = *RHSPtr;

#pragma acc atomic write
  LHS = RHS * 2;

  // expected-error@+2{{statement associated with OpenACC 'atomic write' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be an l-value}}
#pragma acc atomic write
  getRValue<T>() = getRValue<T>();

#pragma acc atomic write
  LHS = getRValue<T>();
}

template<typename T>
void AtomicWriteTemplate2(T LHS, T RHS) {
  // expected-error@+2{{statement associated with OpenACC 'atomic write' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic write
  LHS = RHS;

  T *LHSPtr, *RHSPtr;
#pragma acc atomic write
  LHSPtr = RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic write' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic write
  LHS = *RHSPtr;

#pragma acc atomic write
  LHSPtr = RHS.getPtr();

  // expected-error@+2{{statement associated with OpenACC 'atomic write' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be an l-value}}
#pragma acc atomic write
  getRValue<T>() = getRValue<T>();

  // expected-error@+2{{statement associated with OpenACC 'atomic write' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic write
  LHS = getRValue<T>();
}

void AtomicWrite(int LHS, int RHS) {
  AtomicWriteTemplate(LHS, RHS);
  AtomicWriteTemplate2(S1, S2); // expected-note{{in instantiation of function template specialization}}

#pragma acc atomic write
  LHS = RHS;

  int *LHSPtr, *RHSPtr;
#pragma acc atomic write
  LHSPtr = RHSPtr;

#pragma acc atomic write
  *LHSPtr = *RHSPtr;

  // allowed, expr is ok.
#pragma acc atomic write
  LHS = *RHSPtr;

#pragma acc atomic write
  LHS = RHS * 2;
}

template<typename T>
void AtomicUpdateTemplate(T LHS, T RHS) {
#pragma acc atomic
  LHS++;

#pragma acc atomic update
  LHS--;

#pragma acc atomic
  ++LHS;

#pragma acc atomic update
  --LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic
  +LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic update
  -LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{expected binary operation on right hand side of assignment operator}}
#pragma acc atomic update
  LHS = RHS;

  T *LHSPtr, *RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{expected binary operation on right hand side of assignment operator}}
#pragma acc atomic
  *LHSPtr = *RHSPtr;

  // x binop= expr;
#pragma acc atomic
  LHS += 1 + RHS;
#pragma acc atomic update
  LHS *= 1 + RHS;
#pragma acc atomic
  LHS -= 1 + RHS;
#pragma acc atomic update
  LHS /= 1 + RHS;
#pragma acc atomic
  LHS &= 1 + RHS;
#pragma acc atomic update
  LHS ^= 1 + RHS;
#pragma acc atomic
  LHS |= 1 + RHS;
#pragma acc atomic update
  LHS <<= 1 + RHS;
#pragma acc atomic
  LHS >>= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS != 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS <= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS >= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS %= 1 + RHS;

  // x = x binop expr.
#pragma acc atomic
  LHS = LHS + getRValue<T>();
#pragma acc atomic update
  LHS = LHS * getRValue<T>();
#pragma acc atomic update
  LHS = LHS - getRValue<T>();
#pragma acc atomic update
  LHS = LHS / getRValue<T>();
#pragma acc atomic update
  LHS = LHS & getRValue<T>();
#pragma acc atomic update
  LHS = LHS ^ getRValue<T>();
#pragma acc atomic update
  LHS = LHS | getRValue<T>();
#pragma acc atomic update
  LHS = LHS << getRValue<T>();
#pragma acc atomic update
  LHS = LHS >> getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = LHS < getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = LHS > getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = LHS <= getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = LHS >= getRValue<T>();
#pragma acc atomic update
  LHS = LHS ^ getRValue<T>();


  // x = expr binop x.
#pragma acc atomic
  LHS = getRValue<T>() + LHS;
#pragma acc atomic update
  LHS = getRValue<T>() * LHS;
#pragma acc atomic update
  LHS = getRValue<T>() - LHS;
#pragma acc atomic update
  LHS = getRValue<T>() / LHS;
#pragma acc atomic update
  LHS = getRValue<T>() & LHS;
#pragma acc atomic update
  LHS = getRValue<T>() ^ LHS;
#pragma acc atomic update
  LHS = getRValue<T>() | LHS;
#pragma acc atomic update
  LHS = getRValue<T>() << LHS;
#pragma acc atomic update
  LHS = getRValue<T>() >> LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = getRValue<T>() < LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = getRValue<T>() > LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = getRValue<T>() <= LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic update
  LHS = getRValue<T>() >= LHS;
#pragma acc atomic update
  LHS = getRValue<T>() ^ LHS;

#pragma acc atomic update
  LHS = LHS + getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and 'getRValue<T>()')}}
#pragma acc atomic update
  LHS = RHS + getRValue<T>();

#pragma acc atomic update
  LHS = getRValue<T>() - LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('getRValue<T>()' and 'RHS')}}
#pragma acc atomic update
  LHS = getRValue<T>() + RHS;
}

template<typename T>
void AtomicUpdateTemplate2(T LHS, T RHS) {
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{operand to increment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS++;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{operand to decrement expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS--;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{operand to increment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  ++LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{operand to decrement expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  --LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic
  +LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic update
  -LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic
  LHS();

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic
  *LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{expected binary operation on right hand side of assignment operator}}
#pragma acc atomic update
  LHS = RHS;

  T *LHSPtr, *RHSPtr;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{expected binary operation on right hand side of assignment operator}}
#pragma acc atomic
  *LHSPtr = *RHSPtr;

  // x binop= expr;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS += 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS *= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS -= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS /= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS &= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS |= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS <<= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS >>= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS != 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  LHS %= 1 + RHS;

  // x = x binop expr.
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS = LHS + getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = LHS * getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = LHS - getRValue<T>();

  // x = expr binop x.
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  LHS = getRValue<T>() + LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = getRValue<T>() * LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = getRValue<T>() - LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = LHS + getRValue<T>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and 'getRValue<T>()')}}
#pragma acc atomic update
  LHS = RHS + getRValue<T>();

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  LHS = getRValue<T>() - LHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('getRValue<T>()' and 'RHS')}}
#pragma acc atomic update
  LHS = getRValue<T>() + RHS;
}

void AtomicUpdate() {
  AtomicUpdateTemplate(1, 2);
  AtomicUpdateTemplate2(S1, S2); //expected-note{{in instantiation of function template specialization}}

  int I, J;

#pragma acc atomic
  I++;
#pragma acc atomic update
  --I;
  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{operand to increment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic
  S1++;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{operand to decrement expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  --S2;

  // expected-error@+2{{statement associated with OpenACC 'atomic' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic
  +I;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic update
  -J;

#pragma acc atomic update
  I ^= 1 + J;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  I%= 1 + J;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic update
  S1 ^= 1 + J;

  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic update
  S2 %= 1 + J;

#pragma acc atomic update
  I = I + getRValue<int>();
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('I') must match one side of the sub-operation on the right hand side('J' and 'getRValue<int>()')}}
#pragma acc atomic update
  I = J + getRValue<int>();

#pragma acc atomic update
  I = getRValue<int>() - I;
  // expected-error@+2{{statement associated with OpenACC 'atomic update' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('I') must match one side of the sub-operation on the right hand side('getRValue<int>()' and 'J')}}
#pragma acc atomic update
  I = getRValue<int>() + J;
}

template<typename T>
void AtomicCaptureTemplateSimple(T LHS, T RHS) {
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS++;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
--LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS += 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = RHS;

#pragma acc atomic capture
  LHS = RHS++;

#pragma acc atomic capture
  LHS = RHS--;

#pragma acc atomic capture
  LHS = ++RHS;

#pragma acc atomic capture
  LHS = --RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic capture
  LHS = +RHS;

#pragma acc atomic capture
  LHS = RHS += 1 + RHS;
#pragma acc atomic capture
  LHS = RHS *= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS -= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS /= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS &= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS ^= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS >>= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS |= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS <<= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS >>= 1 + RHS;

#pragma acc atomic capture
  LHS = RHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS <= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS >= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS + 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS < 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS > 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS ^ 1 + RHS;

#pragma acc atomic capture
  LHS = RHS = RHS + 1;
#pragma acc atomic capture
  LHS = RHS = 1 + RHS;
#pragma acc atomic capture
  LHS = RHS = RHS * 1;
#pragma acc atomic capture
  LHS = RHS = 1 * RHS;
#pragma acc atomic capture
  LHS = RHS = RHS / 1;
#pragma acc atomic capture
  LHS = RHS = 1 / RHS;
#pragma acc atomic capture
  LHS = RHS = RHS ^ 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = 1 % RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = RHS < 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = 1 > RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and 'getRValue<T>()')}}
#pragma acc atomic capture
  LHS = LHS = RHS + getRValue<T>();

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('getRValue<T>()' and 'RHS')}}
#pragma acc atomic capture
  LHS = LHS = getRValue<T>() + RHS;
}
template<typename T>
void AtomicCaptureTemplateSimple2(T LHS, T RHS) {
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS++;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
--LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS += 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS++;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS--;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = ++RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = --RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic capture
  LHS = +RHS;


  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS += 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS *= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS -= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS /= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS &= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS >>= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS |= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS <<= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS >>= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS <= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS >= 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS + 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS < 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS > 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS ^ 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = RHS + 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = RHS * 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = 1 * RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = RHS / 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = 1 / RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  LHS = RHS = RHS ^ 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = 1 % RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = RHS < 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = 1 > RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and 'getRValue<T>()')}}
#pragma acc atomic capture
  LHS = LHS = RHS + getRValue<T>();

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('getRValue<T>()' and 'RHS')}}
#pragma acc atomic capture
  LHS = LHS = getRValue<T>() + RHS;
}

void AtomicCaptureSimple(int LHS, int RHS) {
  AtomicCaptureTemplateSimple(1, 2);
  AtomicCaptureTemplateSimple2(S1, S2); //expected-note{{in instantiation of function template specialization}}

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS++;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
--LHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment expression}}
#pragma acc atomic capture
  LHS += 1 + RHS;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = RHS;

#pragma acc atomic capture
  LHS = RHS++;

#pragma acc atomic capture
  LHS = RHS--;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = ++S2;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = --S2 ;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{unary operator not supported, only increment and decrement operations permitted}}
#pragma acc atomic capture
  LHS = +RHS;

#pragma acc atomic capture
  LHS = RHS += 1 + RHS;
#pragma acc atomic capture
  LHS = RHS *= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = RHS -= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = S1 /= 1 + RHS;
#pragma acc atomic capture
  LHS = RHS &= 1 + S2;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = S1^= 1 + S2;

#pragma acc atomic capture
  LHS = RHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS <= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = RHS ^= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = S1 <= 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{compound binary operator not supported, only +=, *=, -=, /=, &=, ^=, |=, <<=, or >>= are permitted}}
#pragma acc atomic capture
  LHS = RHS <= 1 + S2;

#pragma acc atomic capture
  LHS = RHS = RHS + 1;
#pragma acc atomic capture
  LHS = RHS = 1 + RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = RHS = RHS * 1;
  // A little weird, because this contains a 'operator int' call here rather
  // than a conversion, so the diagnostic could be better.
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = S2 = 1 * S2;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = RHS < 1;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{binary operator not supported, only +, *, -, /, &, ^, |, <<, or >> are permitted}}
#pragma acc atomic capture
  LHS = RHS = 1 > RHS;
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left operand to assignment expression must be of scalar type (was 'Struct')}}
#pragma acc atomic capture
  S1 = RHS = RHS < 1;

  // A little weird, because this contains a 'operator int' call here rather
  // than a conversion, so the diagnostic could be better.
  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  LHS = S1 = 1 > S1;

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and 'getRValue<int>()')}}
#pragma acc atomic capture
  LHS = LHS = RHS + getRValue<int>();

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('getRValue<int>()' and 'RHS')}}
#pragma acc atomic capture
  LHS = LHS = getRValue<int>() + RHS;
}

template<typename T>
void AtomicCaptureTemplateCompound(T LHS, T RHS) {

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+4{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
    LHS = RHS;
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+3{{'atomic capture' with a compound statement only supports two statements}}
#pragma acc atomic capture
  {
    LHS = RHS; RHS += 1; LHS=RHS;
  }


#pragma acc atomic capture
  {
    LHS++;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    ++LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    --LHS;
    RHS = LHS;
  }


#pragma acc atomic capture
  {
    LHS--;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used in unary expression('LHS') from the first statement}}
    LHS = RHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    --LHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x binop = expr; v = x; }
#pragma acc atomic capture
  {
    LHS += 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS *= 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of compound assignment('LHS') from the first statement}}
    LHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS /= 1;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = x binop expr; v = x; }
#pragma acc atomic capture
  {
    LHS = LHS + 1;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
  // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS - 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS * 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS / 1;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = expr binop x; v = x; }
#pragma acc atomic capture
  {
    LHS = 1 ^ LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 & RHS;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS | 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS << 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { v = x; x binop = expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS += 1;
  }

  // { v = x; x = x binop expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = RHS / 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('LHS' and '1')}}
    RHS = LHS ^ 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS << 1;
  }
  // { v = x; x = expr binop x; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = 1 / RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('1' and 'LHS')}}
    RHS = 1 ^ LHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 << RHS;
  }

  // { v = x; x = expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on left hand side of assignment('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS = 1;
  }

  // { v = x; x++; }
  // { v = x; ++x; }
  // { v = x; x--; }
  // { v = x; --x; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS++;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS--;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    ++RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    --RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable in unary expression('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS++;
  }
}

template<typename T>
void AtomicCaptureTemplateCompound2(T LHS, T RHS) {

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+4{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
    LHS = RHS;
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+3{{'atomic capture' with a compound statement only supports two statements}}
#pragma acc atomic capture
  {
    LHS = RHS; RHS += 1; LHS=RHS;
  }


#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{operand to increment expression must be of scalar type (was 'Struct')}}
    LHS++;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{operand to increment expression must be of scalar type (was 'Struct')}}
    ++LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{operand to decrement expression must be of scalar type (was 'Struct')}}
    --LHS;
    RHS = LHS;
  }


#pragma acc atomic capture
  {
    LHS--;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used in unary expression('LHS') from the first statement}}
    LHS = RHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    --LHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x binop = expr; v = x; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
    LHS += 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS *= 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of compound assignment('LHS') from the first statement}}
    LHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS /= 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = x binop expr; v = x; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = LHS + 1;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
  // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS - 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS * 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS / 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = expr binop x; v = x; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = 1 ^ LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
  // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 & RHS;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS | 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS << 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { v = x; x binop = expr; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS += 1;
  }

  // { v = x; x = x binop expr; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS = RHS / 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('LHS' and '1')}}
    RHS = LHS ^ 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS << 1;
  }
  // { v = x; x = expr binop x; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS = 1 / RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('1' and 'LHS')}}
    RHS = 1 ^ LHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 << RHS;
  }

  // { v = x; x = expr; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS = 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on left hand side of assignment('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS = 1;
  }

  // { v = x; x++; }
  // { v = x; ++x; }
  // { v = x; x--; }
  // { v = x; --x; }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS++;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    RHS--;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    ++RHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    LHS = RHS;
    --RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable in unary expression('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS++;
  }
}
void AtomicCaptureCompound(int LHS, int RHS) {
  AtomicCaptureTemplateCompound(1, 2); 
  AtomicCaptureTemplateCompound2(S1, S2); //expected-note{{in instantiation of function template specialization}}

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+2{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+4{{expected assignment, compound assignment, increment, or decrement expression}}
#pragma acc atomic capture
  {
    LHS = RHS;
  }

  // expected-error@+2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+3{{'atomic capture' with a compound statement only supports two statements}}
#pragma acc atomic capture
  {
    LHS = RHS; RHS += 1; LHS=RHS;
  }


#pragma acc atomic capture
  {
    LHS++;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{operand to increment expression must be of scalar type (was 'Struct')}}
    S1++;
    S2= S1;
  }

#pragma acc atomic capture
  {
    ++LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    --LHS;
    RHS = LHS;
  }


#pragma acc atomic capture
  {
    LHS--;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used in unary expression('LHS') from the first statement}}
    LHS = RHS;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -LHS;
    RHS = LHS;
  }

#pragma acc atomic capture
  {
    --LHS;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x binop = expr; v = x; }
#pragma acc atomic capture
  {
    LHS += 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to compound assignment expression must be of scalar type (was 'Struct')}}
    S1 += 1;
    S2= S1;
  }
#pragma acc atomic capture
  {
    LHS *= 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of compound assignment('LHS') from the first statement}}
    LHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS /= 1;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = x binop expr; v = x; }
#pragma acc atomic capture
  {
    LHS = LHS + 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    S1 = S1 + 1;
    S2= S1;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS - 1;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS * 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS / 1;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { x = expr binop x; v = x; }
#pragma acc atomic capture
  {
    LHS = 1 ^ LHS;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    S1 = 1 ^ S1;
    S2 = S1;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 & RHS;
    RHS = LHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS | 1;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on right hand side of assignment('RHS') must match variable used on left hand side of assignment('LHS') from the first statement}}
    RHS = RHS;
  }
#pragma acc atomic capture
  {
    LHS = LHS << 1;
  // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
  // expected-note@+1{{expected assignment expression}}
    RHS += LHS;
  }

  // { v = x; x binop = expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS += 1;
  }

#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    S1 = S2;
    S2 += 1;
  }

  // { v = x; x = x binop expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = RHS / 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('LHS' and '1')}}
    RHS = LHS ^ 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('RHS' and '1')}}
    LHS = RHS << 1;
  }
  // { v = x; x = expr binop x; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = 1 / RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('RHS') must match one side of the sub-operation on the right hand side('1' and 'LHS')}}
    RHS = 1 ^ LHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left hand side of assignment operation('LHS') must match one side of the sub-operation on the right hand side('1' and 'RHS')}}
    LHS = 1 << RHS;
  }

  // { v = x; x = expr; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS = 1;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable on left hand side of assignment('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS = 1;
  }

  // { v = x; x++; }
  // { v = x; ++x; }
  // { v = x; x--; }
  // { v = x; --x; }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS++;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    RHS--;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    ++RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    --RHS;
  }
#pragma acc atomic capture
  {
    // expected-error@-2{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{left operand to assignment expression must be of scalar type (was 'Struct')}}
    S1= S2;
    --S2;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{unary operator not supported, only increment and decrement operations permitted}}
    -RHS;
  }
#pragma acc atomic capture
  {
    LHS = RHS;
    // expected-error@-3{{statement associated with OpenACC 'atomic capture' directive is invalid}}
    // expected-note@+1{{variable in unary expression('LHS') must match variable used on right hand side of assignment('RHS') from the first statement}}
    LHS++;
  }
}
