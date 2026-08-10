// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify %s -fblocks

int (*FP)();
int (^IFP) ();
int (^II) (int);
int main(void) {
  int (*FPL) (int) = FP; // C doesn't consider this an error.

  // For Blocks, the ASTContext::typesAreBlockCompatible() makes sure this is an error.
  int (^PFR) (int) = IFP; // OK
  PFR = II;       // OK

  int (^IFP) () = PFR; // OK


  const int (^CIC) () = IFP; // OK -  initializing 'const int (^)()' with an expression of type 'int (^)()'}}

  const int (^CICC) () = CIC;


  int * const (^IPCC) () = 0;

  int * const (^IPCC1) () = IPCC;

  int * (^IPCC2) () = IPCC;       // expected-error {{incompatible block pointer types initializing 'int *(^)()' with an expression of type 'int *const (^)()'}}

  int (^IPCC3) (const int) = PFR;

  int (^IPCC4) (int, char (^CArg) (double));

  int (^IPCC5) (int, char (^CArg) (double)) = IPCC4;

  int (^IPCC6) (int, char (^CArg) (float))  = IPCC4; // expected-error {{incompatible block pointer types initializing 'int (^)(int, char (^)(float))' with an expression of type 'int (^)(int, char (^)(double))'}}

  IPCC2 = 0;
  IPCC1 = 1; // expected-error {{invalid block pointer conversion assigning to 'int *const (^)()' from 'int'}}
  int (^x)() = 0;
  int (^y)() = 3;   // expected-error {{invalid block pointer conversion initializing 'int (^)()' with an expression of type 'int'}}
  int a = 1;
  int (^z)() = a+4;   // expected-error {{invalid block pointer conversion initializing 'int (^)()' with an expression of type 'int'}}
}

int blah(void) {
  int (^IFP) (float);
  char (^PCP)(double, double, char);

  IFP(1.0);
  IFP (1.0, 2.0); // expected-error {{too many arguments to block call}}

  char ch = PCP(1.0, 2.0, 'a');
  return PCP(1.0, 2.0);   // expected-error {{too few arguments to block}}
}
