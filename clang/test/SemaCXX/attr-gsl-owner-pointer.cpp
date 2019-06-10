// RUN: %clang_cc1 -fsyntax-only -verify %s

int [[gsl::Owner]] i;
//expected-error@-1 {{'Owner' attribute cannot be applied to types}}
void [[gsl::Owner]] f();
//expected-error@-1 {{'Owner' attribute cannot be applied to types}}

struct S {
};

S [[gsl::Owner]] Instance;
//expected-error@-1 {{'Owner' attribute cannot be applied to types}}

class [[gsl::Owner]] OwnerMissingParameter {};
//expected-error@-1 {{'Owner' attribute takes one argument}}
class [[gsl::Pointer]] PointerMissingParameter {};
//expected-error@-1 {{'Pointer' attribute takes one argument}}


class [[gsl::Owner(7)]] OwnerDerefNoType {};
class [[gsl::Pointer("int")]] PointerDerefNoType {};


class [[gsl::Owner(int)]] [[gsl::Pointer(int)]] BothOwnerPointer {};

class [[gsl::Owner(int)]] [[gsl::Owner(int)]] DuplicateOwner {};
class [[gsl::Pointer(int)]] [[gsl::Pointer(int)]] DuplicatePointer {};

class [[gsl::Owner(int)]] [[gsl::Owner(char)]] OwnerDifferentTypes {};
class [[gsl::Pointer(int)]] [[gsl::Pointer(char)]] PointerDifferentTypes {};

class [[gsl::Owner(void)]] OwnerVoidDerefType {};
class [[gsl::Pointer(void)]] PointerVoidDerefType {};

class [[gsl::Owner(int)]] AnOwner {};
class [[gsl::Pointer(S)]] APointer {};
