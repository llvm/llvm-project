// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fsyntax-only -verify %s
template<typename T> int force_same(T, T);

// C++ [dcl.enum]p5:
//   [...] If the underlying type is not fixed, the type of each enumerator is 
//   the type of its initializing value:
//     - If an initializer is specified for an enumerator, the initializing 
//       value has the same type as the expression.
enum Bullet1 {
  Bullet1Val1 = 'a',
  Bullet1Val2 = 10u,
  Bullet1Val1IsChar = sizeof(force_same(Bullet1Val1, char(0))),
  Bullet1Val2IsUnsigned = sizeof(force_same(Bullet1Val2, unsigned(0)))
};

//    - If no initializer is specified for the first enumerator, the 
//      initializing value has an unspecified integral type.
enum Bullet2 {
  Bullet2Val,
  Bullet2ValIsInt = sizeof(force_same(Bullet2Val, int(0)))
};

//    - Otherwise the type of the initializing value is the same as the type
//      of the initializing value of the preceding enumerator unless the 
//      incremented value is not representable in that type, in which case the
//      type is an unspecified integral type sufficient to contain the 
//      incremented value. If no such type exists, the program is ill-formed.
enum Bullet3a {
  Bullet3aVal1 = 17,
  Bullet3aVal2,
  Bullet3aVal2IsInt = sizeof(force_same(Bullet3aVal2, int(0))),
  Bullet3aVal3 = 2147483647,
  Bullet3aVal3IsInt = sizeof(force_same(Bullet3aVal3, int(0))),
  Bullet3aVal4,
  Bullet3aVal4IsUnsigned = sizeof(force_same(Bullet3aVal4, 0ul))
};

enum Bullet3b {
  Bullet3bVal1 = 17u,
  Bullet3bVal2,
  Bullet3bVal2IsInt = sizeof(force_same(Bullet3bVal2, 0u)),
  Bullet3bVal3 = 2147483647u,
  Bullet3bVal3IsInt = sizeof(force_same(Bullet3bVal3, 0u)),
  Bullet3bVal4,
  Bullet3bVal4IsUnsigned = sizeof(force_same(Bullet3bVal4, 0ul))
};

enum Bullet3c {
  Bullet3cVal1 = 0xFFFFFFFFFFFFFFFEull,
  Bullet3cVal2,
  Bullet3cVal3 // expected-warning{{not representable}}
};

//   Following the closing brace of an enum-specifier, each enumerator has the
//   type of its enumeration.
int array0[sizeof(force_same(Bullet3bVal3, Bullet3b(0)))? 1 : -1];
