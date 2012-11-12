// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s -fblocks

int (*FP)();
int (^IFP) ();
int (^II) (int);
int main() {
  int (*FPL) (int) = FP; // expected-error {{cannot initialize a variable of type 'int (*)(int)' with an lvalue of type 'int (*)()'}} 

  // For Blocks, the ASTContext::typesAreBlockCompatible() makes sure this is an error.
  int (^PFR) (int) = IFP; // expected-error {{cannot initialize a variable of type 'int (^)(int)' with an lvalue of type 'int (^)()'}}
  PFR = II;       // OK

  int (^IFP) () = PFR; // OK


  const int (^CIC) () = IFP; // OK -  initializing 'const int (^)()' with an expression of type 'int (^)()'}}

  const int (^CICC) () = CIC;


  int * const (^IPCC) () = 0;

  int * const (^IPCC1) () = IPCC;

  int * (^IPCC2) () = IPCC;       // expected-error  {{cannot initialize a variable of type 'int *(^)()' with an lvalue of type 'int *const (^)()'}}

  int (^IPCC3) (const int) = PFR;

  int (^IPCC4) (int, char (^CArg) (double));

  int (^IPCC5) (int, char (^CArg) (double)) = IPCC4;

  int (^IPCC6) (int, char (^CArg) (float))  = IPCC4; // expected-error {{cannot initialize a variable of type 'int (^)(int, char (^)(float))' with an lvalue of type}}

  IPCC2 = 0;
  IPCC2 = 1; 
  int (^x)() = 0;
  int (^y)() = 3;   // expected-error {{cannot initialize a variable of type 'int (^)()' with an rvalue of type 'int'}}
  int a = 1;
  int (^z)() = a+4;   // expected-error {{cannot initialize a variable of type 'int (^)()' with an rvalue of type 'int'}}
}

int blah() {
  int (^IFP) (float);
  char (^PCP)(double, double, char);

  IFP(1.0);
  IFP (1.0, 2.0); // expected-error {{too many arguments to block call}}

  char ch = PCP(1.0, 2.0, 'a');
  return PCP(1.0, 2.0);   // expected-error {{too few arguments to block}}
}
