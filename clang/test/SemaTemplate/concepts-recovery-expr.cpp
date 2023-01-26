// RUN: %clang_cc1 -std=c++20 -verify %s

// expected-error@+1{{use of undeclared identifier 'b'}}
constexpr bool CausesRecoveryExpr = b;

template<typename T>
concept ReferencesCRE = CausesRecoveryExpr;

template<typename T> requires CausesRecoveryExpr // #NVC1REQ
void NoViableCands1(){} // #NVC1

template<typename T> requires ReferencesCRE<T> // #NVC2REQ
void NoViableCands2(){} // #NVC2

template<ReferencesCRE T> // #NVC3REQ
void NoViableCands3(){} // #NVC3

void NVCUse() {
  NoViableCands1<int>();
  // expected-error@-1 {{no matching function for call to 'NoViableCands1'}}
  // expected-note@#NVC1{{candidate template ignored: constraints not satisfied}}
  // expected-note@#NVC1REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

  NoViableCands2<int>();
  // expected-error@-1 {{no matching function for call to 'NoViableCands2'}}
  // expected-note@#NVC2{{candidate template ignored: constraints not satisfied}}
  // expected-note@#NVC2REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  NoViableCands3<int>();
  // expected-error@-1 {{no matching function for call to 'NoViableCands3'}}
  // expected-note@#NVC3{{candidate template ignored: constraints not satisfied}}
  // expected-note@#NVC3REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
}

template<typename T> requires CausesRecoveryExpr // #OVC1REQ
void OtherViableCands1(){} // #OVC1

template<typename T>
void OtherViableCands1(){} // #OVC1_ALT

template<typename T> requires ReferencesCRE<T> // #OVC2REQ
void OtherViableCands2(){} // #OVC2

template<typename T>
void OtherViableCands2(){} // #OVC2_ALT

template<ReferencesCRE T> // #OVC3REQ
void OtherViableCands3(){} // #OVC3
template<typename T>
void OtherViableCands3(){} // #OVC3_ALT

void OVCUse() {
  OtherViableCands1<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands1'}}
  // expected-note@#OVC1_ALT {{candidate function}}
  // expected-note@#OVC1 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OVC1REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  OtherViableCands2<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands2'}}
  // expected-note@#OVC2_ALT {{candidate function}}
  // expected-note@#OVC2 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OVC2REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  OtherViableCands3<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands3'}}
  // expected-note@#OVC3_ALT {{candidate function}}
  // expected-note@#OVC3 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OVC3REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
}

template<typename T> requires CausesRecoveryExpr // #OBNVC1REQ
void OtherBadNoViableCands1(){} // #OBNVC1

template<typename T> requires false // #OBNVC1REQ_ALT
void OtherBadNoViableCands1(){} // #OBNVC1_ALT

template<typename T> requires ReferencesCRE<T> // #OBNVC2REQ
void OtherBadNoViableCands2(){} // #OBNVC2

template<typename T> requires false// #OBNVC2REQ_ALT
void OtherBadNoViableCands2(){} // #OBNVC2_ALT

template<ReferencesCRE T> // #OBNVC3REQ
void OtherBadNoViableCands3(){} // #OBNVC3
template<typename T> requires false // #OBNVC3REQ_ALT
void OtherBadNoViableCands3(){} // #OBNVC3_ALT

void OBNVCUse() {
  OtherBadNoViableCands1<int>();
  // expected-error@-1 {{no matching function for call to 'OtherBadNoViableCands1'}}
  // expected-note@#OBNVC1_ALT {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC1REQ_ALT {{because 'false' evaluated to false}}
  // expected-note@#OBNVC1 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC1REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  OtherBadNoViableCands2<int>();
  // expected-error@-1 {{no matching function for call to 'OtherBadNoViableCands2'}}
  // expected-note@#OBNVC2_ALT {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC2REQ_ALT {{because 'false' evaluated to false}}
  // expected-note@#OBNVC2 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC2REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  OtherBadNoViableCands3<int>();
  // expected-error@-1 {{no matching function for call to 'OtherBadNoViableCands3'}}
  // expected-note@#OBNVC3_ALT {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC3REQ_ALT {{because 'false' evaluated to false}}
  // expected-note@#OBNVC3 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#OBNVC3REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
}


// Same tests with member functions.
struct OVC {
template<typename T> requires CausesRecoveryExpr // #MEMOVC1REQ
void OtherViableCands1(){} // #MEMOVC1

template<typename T>
void OtherViableCands1(){} // #MEMOVC1_ALT

template<typename T> requires ReferencesCRE<T> // #MEMOVC2REQ
void OtherViableCands2(){} // #MEMOVC2

template<typename T>
void OtherViableCands2(){} // #MEMOVC2_ALT

template<ReferencesCRE T> // #MEMOVC3REQ
void OtherViableCands3(){} // #MEMOVC3
template<typename T>
void OtherViableCands3(){} // #MEMOVC3_ALT
};

void MemOVCUse() {
  OVC S;
  S.OtherViableCands1<int>();
  // expected-error@-1 {{no matching member function for call to 'OtherViableCands1'}}
  // expected-note@#MEMOVC1_ALT {{candidate function}}
  // expected-note@#MEMOVC1 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#MEMOVC1REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  S.OtherViableCands2<int>();
  // expected-error@-1 {{no matching member function for call to 'OtherViableCands2'}}
  // expected-note@#MEMOVC2_ALT {{candidate function}}
  // expected-note@#MEMOVC2 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#MEMOVC2REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  S.OtherViableCands3<int>();
  // expected-error@-1 {{no matching member function for call to 'OtherViableCands3'}}
  // expected-note@#MEMOVC3_ALT {{candidate function}}
  // expected-note@#MEMOVC3 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#MEMOVC3REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
}

struct StaticOVC {
template<typename T> requires CausesRecoveryExpr // #SMEMOVC1REQ
static void OtherViableCands1(){} // #SMEMOVC1

template<typename T>
static void OtherViableCands1(){} // #SMEMOVC1_ALT

template<typename T> requires ReferencesCRE<T> // #SMEMOVC2REQ
static void OtherViableCands2(){} // #SMEMOVC2

template<typename T>
static void OtherViableCands2(){} // #SMEMOVC2_ALT

template<ReferencesCRE T> // #SMEMOVC3REQ
static void OtherViableCands3(){} // #SMEMOVC3
template<typename T>
static void OtherViableCands3(){} // #SMEMOVC3_ALT
};

void StaticMemOVCUse() {
  StaticOVC::OtherViableCands1<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands1'}}
  // expected-note@#SMEMOVC1_ALT {{candidate function}}
  // expected-note@#SMEMOVC1 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#SMEMOVC1REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  StaticOVC::OtherViableCands2<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands2'}}
  // expected-note@#SMEMOVC2_ALT {{candidate function}}
  // expected-note@#SMEMOVC2 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#SMEMOVC2REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
  StaticOVC::OtherViableCands3<int>();
  // expected-error@-1 {{no matching function for call to 'OtherViableCands3'}}
  // expected-note@#SMEMOVC3_ALT {{candidate function}}
  // expected-note@#SMEMOVC3 {{candidate template ignored: constraints not satisfied}}
  // expected-note@#SMEMOVC3REQ{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}
}
