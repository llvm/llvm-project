// RUN: %clang_cc1 -fsyntax-only -verify -Wswitch-enum -Wcovered-switch-default -triple x86_64-linux-gnu %s
void f (int z) { 
  while (z) { 
    default: z--;            // expected-error {{statement not in switch}}
  } 
}

void foo(int X) {
  switch (X) {
  case 42: ;                 // expected-note {{previous case}}
  case 5000000000LL:         // expected-warning {{overflow}}
  case 42:                   // expected-error {{duplicate case value '42'}}
   ;

  case 100 ... 99: ;         // expected-warning {{empty case range}}

  case 43: ;                 // expected-note {{previous case}}
  case 43 ... 45:  ;         // expected-error {{duplicate case value}}

  case 100 ... 20000:;       // expected-note {{previous case}}
  case 15000 ... 40000000:;  // expected-error {{duplicate case value}}
  }
}

void test3(void) { 
  // empty switch;
  switch (0); // expected-warning {{no case matching constant switch condition '0'}} \
              // expected-warning {{switch statement has empty body}} \
              // expected-note{{put the semicolon on a separate line to silence this warning}}
}

extern int g();

void test4()
{
  int cond;
  switch (cond) {
  case 0 && g():
  case 1 || g():
    break;
  }

  switch(cond)  {
  case g(): // expected-error {{expression is not an integer constant expression}}
  case 0 ... g(): // expected-error {{expression is not an integer constant expression}}
    break;
  }
  
  switch (cond) {
  case 0 && g() ... 1 || g():
    break;
  }
  
  switch (cond) {
  case g() // expected-error {{expression is not an integer constant expression}}
      && 0:
    break;
  }
  
  switch (cond) {
  case 0 ...
      g() // expected-error {{expression is not an integer constant expression}}
      || 1:
    break;
  }
}

void test5(int z) { 
  switch(z) {
    default:  // expected-note {{previous case defined here}}
    default:  // expected-error {{multiple default labels in one switch}}
      break;
  }
} 

void test6() {
  char ch = 'a';
  switch(ch) {
    case 1234:  // expected-warning {{overflow converting case value}}
      break;
  }
}

// PR5606
int f0(int var) {
  switch (va) { // expected-error{{use of undeclared identifier 'va'}}
  case 1:
    break;
  case 2:
    return 1;
  }
  return 2;
}

void test7() {
  enum {
    A = 1,
    B
  } a;
  switch(a) { //expected-warning{{enumeration value 'B' not handled in switch}}
    case A:
      break;
  }
  switch(a) {
    case B:
    case A:
      break;
  }
  switch(a) {
    case A:
    case B:
    case 3: // expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
  switch(a) {
    case A:
    case B:
    case 3 ... //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
        4: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
  switch(a) {
    case 1 ... 2:
      break;
  }
  switch(a) {
    case 0 ... 2: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
  switch(a) {
    case 1 ... 3: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
  switch(a) {
    case 0 ...  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      3:  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }

}

void test8() {
  enum {
    A,
    B,
    C = 1
  } a;
  switch(a) {
    case A:
    case B:
     break;
  }
  switch(a) {
    case A:
    case C:
      break;
  }
  switch(a) { //expected-warning{{enumeration value 'B' not handled in switch}}
    case A:
      break;
  }
}

void test9() {
  enum {
    A = 3,
    C = 1
  } a;
  switch(a) {
    case 0: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
    case 1:
    case 2: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
    case 3:
    case 4: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
}

void test10() {
  enum {
    A = 10,
    C = 2,
    B = 4,
    D = 12
  } a;
  switch(a) {
    case 0 ...  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
	    1:  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
    case 2 ... 4:
    case 5 ...  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}	
	      9:  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
    case 10 ... 12:
    case 13 ...  //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
              16: //expected-warning{{case value not in enumerated type 'enum (anonymous enum}}
      break;
  }
}

void test11() {
  enum {
    A = -1,
    B,
    C
  } a;
  switch(a) { //expected-warning{{enumeration value 'A' not handled in switch}}
    case B:
    case C:
      break;
  }

  switch(a) { //expected-warning{{enumeration value 'A' not explicitly handled in switch}}
    case B:
    case C:
      break;
      
    default:
      break;
  }
}

void test12() {
  enum {
    A = -1,
    B = 4294967286
  } a;
  switch(a) {
    case A:
    case B:
      break;
  }
}

// <rdar://problem/7643909>
typedef enum {
    val1,
    val2,
    val3
} my_type_t;

int test13(my_type_t t) {
  switch(t) { // expected-warning{{enumeration value 'val3' not handled in switch}}
  case val1:
    return 1;
  case val2:
    return 2;
  }
  return -1;
}

// <rdar://problem/7658121>
enum {
  EC0 = 0xFFFF0000,
  EC1 = 0xFFFF0001,
};

int test14(int a) {
  switch(a) {
  case EC0: return 0;
  case EC1: return 1;
  }
  return 0;
}

void f1(unsigned x) {
  switch (x) {
  case -1: break;
  default: break;
  }
}

void test15() {
  int i = 0;
  switch (1) { // expected-warning {{no case matching constant switch condition '1'}}
  case 0: i = 0; break;
  case 2: i++; break;
  }
}

void test16() {
  const char c = '5';
  switch (c) { // expected-warning {{no case matching constant switch condition '53'}}
  case '6': return;
  }
}

// PR7359
void test17(int x) {
  switch (x >= 17) { // expected-warning {{switch condition has boolean value}}
  case 0: return;
  }

  switch ((int) (x <= 17)) {
  case 0: return;
  }
}

int test18() {
  enum { A, B } a;
  switch (a) {
  case A: return 0;
  case B: return 1;
  case 7: return 1; // expected-warning {{case value not in enumerated type}}
  default: return 2; // expected-warning {{default label in switch which covers all enumeration values}}
  }
}

// rdar://110822110
typedef enum {
        kOne = 1,
} Ints;
        
void rdar110822110(Ints i)
{
        switch (i) {
                case kOne:
                        break;
                case 2: 	// expected-warning {{case value not in enumerated type 'Ints'}}          
                        break;
                default:	// expected-warning {{default label in switch which covers all enumeration values}}
                        break;
                }
}

// PR9243
#define TEST19MACRO 5
void test19(int i) {
  enum {
    kTest19Enum1 = 7,
    kTest19Enum2 = kTest19Enum1
  };
  const int a = 3;
  switch (i) {
    case 5: // expected-note {{previous case}}
    case TEST19MACRO: // expected-error {{duplicate case value '5'}}

    case 7: // expected-note {{previous case}}
    case kTest19Enum1: // expected-error {{duplicate case value: '7' and 'kTest19Enum1' both equal '7'}} \
                       // expected-note {{previous case}}
    case kTest19Enum1: // expected-error {{duplicate case value 'kTest19Enum1'}} \
                       // expected-note {{previous case}}
    case kTest19Enum2: // expected-error {{duplicate case value: 'kTest19Enum1' and 'kTest19Enum2' both equal '7'}} \
                       // expected-note {{previous case}}
    case (int)kTest19Enum2: //expected-error {{duplicate case value 'kTest19Enum2'}}

    case 3: // expected-note {{previous case}}
    case a: // expected-error {{duplicate case value: '3' and 'a' both equal '3'}} \
            // expected-note {{previous case}}
    case a: // expected-error {{duplicate case value 'a'}}
      break;
  }
}

// Allow the warning 'case value not in enumerated type' to be silenced with
// the following pattern.
//
// If 'case' expression refers to a static const variable of the correct enum
// type, then we count this as a sufficient declaration of intent by the user,
// so we silence the warning.
enum ExtendedEnum1 {
  EE1_a,
  EE1_b
};

enum ExtendedEnum1_unrelated { EE1_misc };

static const enum ExtendedEnum1 EE1_c = 100;
static const enum ExtendedEnum1_unrelated EE1_d = 101;

void switch_on_ExtendedEnum1(enum ExtendedEnum1 e) {
  switch(e) {
  case EE1_a: break;
  case EE1_b: break;
  case EE1_c: break; // no-warning
  case EE1_d: break; // expected-warning {{case value not in enumerated type 'enum ExtendedEnum1'}}
  // expected-warning@-1 {{comparison of two values with different enumeration types in switch statement ('enum ExtendedEnum1' and 'enum ExtendedEnum1_unrelated')}}
  }
}

void PR11778(char c, int n, long long ll) {
  // Do not reject this; we don't have duplicate case values because we
  // check for duplicates in the promoted type.
  switch (c) case 1: case 257: ; // expected-warning {{overflow}}

  switch (n) case 0x100000001LL: case 1: ; // expected-warning {{overflow}} expected-error {{duplicate}} expected-note {{previous}}
  switch ((int)ll) case 0x100000001LL: case 1: ; // expected-warning {{overflow}} expected-error {{duplicate}} expected-note {{previous}}
  switch ((long long)n) case 0x100000001LL: case 1: ;
  switch (ll) case 0x100000001LL: case 1: ;
}
