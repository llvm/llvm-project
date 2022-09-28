// RUN: %clang_cc1 -verify -std=c2x %s

/* WG14 N2927: yes
 * Not-so-magic: typeof
 */

// These examples originated in WG14 N2927 but were modified to test particular
// compiler behaviors. Each of these examples come from C2x 6.7.2.5.

// EXAMPLE 1
typeof(1 + 1) func();
int func();

// EXAMPLE 2
const _Atomic int purr = 0;
const int meow = 1;
const char *const mew[] = {
	"aardvark",
	"bluejay",
	"catte",
};

extern typeof_unqual(purr) plain_purr;
extern int plain_purr;

extern typeof(_Atomic typeof(meow)) atomic_meow;
extern const _Atomic int atomic_meow;

extern typeof(mew) mew_array;
extern const char *const mew_array[3];

extern typeof_unqual(mew) mew2_array;
extern const char *mew2_array[3];

// EXAMPLE 3
void foo(int argc, char *argv[]) { // expected-note 2 {{declared here}}
  _Static_assert(sizeof(typeof('p')) == sizeof(int));
  _Static_assert(sizeof(typeof('p')) == sizeof('p'));
  _Static_assert(sizeof(typeof((char)'p')) == sizeof(char));
  _Static_assert(sizeof(typeof((char)'p')) == sizeof((char)'p'));
  _Static_assert(sizeof(typeof("meow")) == sizeof(char[5]));
  _Static_assert(sizeof(typeof("meow")) == sizeof("meow"));
  _Static_assert(sizeof(typeof(argc)) == sizeof(int));
  _Static_assert(sizeof(typeof(argc)) == sizeof(argc));
  _Static_assert(sizeof(typeof(argv)) == sizeof(char**));
  _Static_assert(sizeof(typeof(argv)) == sizeof(argv)); // expected-warning {{sizeof on array function parameter will return size of 'char **' instead of 'char *[]'}}

  _Static_assert(sizeof(typeof_unqual('p')) == sizeof(int));
  _Static_assert(sizeof(typeof_unqual('p')) == sizeof('p'));
  _Static_assert(sizeof(typeof_unqual((char)'p')) == sizeof(char));
  _Static_assert(sizeof(typeof_unqual((char)'p')) == sizeof((char)'p'));
  _Static_assert(sizeof(typeof_unqual("meow")) == sizeof(char[5]));
  _Static_assert(sizeof(typeof_unqual("meow")) == sizeof("meow"));
  _Static_assert(sizeof(typeof_unqual(argc)) == sizeof(int));
  _Static_assert(sizeof(typeof_unqual(argc)) == sizeof(argc));
  _Static_assert(sizeof(typeof_unqual(argv)) == sizeof(char**));
  _Static_assert(sizeof(typeof_unqual(argv)) == sizeof(argv)); // expected-warning {{sizeof on array function parameter will return size of 'char **' instead of 'char *[]'}}
}

// EXAMPLE 4
void bar(int argc) {
  extern int val;
  extern typeof(typeof_unqual(typeof(argc)))val;
}

// EXAMPLE 5 is tested by n2927_2.c because it is a codegen test.

// EXAMPLE 6
extern const char *y[4];
extern typeof(typeof(const char*)[4]) y;

// EXAMPLE 7
void f(int);

void g(double);
typeof(f(5)) g(double x);          // g has type "void(double)"

extern void (*h)(double);
extern typeof(g)* h;               // h has type "void(*)(double)"
extern typeof(true ? g : 0) h;  // h has type "void(*)(double)"

void j(double *, double **);
void j(double A[5], typeof(A)* B); // j has type "void(double*, double**)"

extern typeof(double[]) D;         // D has an incomplete type

extern double C[2];
extern typeof(D) C;                // C has type "double[2]"

typeof(D) D = { 5, 8.9, 0.1, 99 }; // D is now completed to "double[4]"
extern double E[4];
extern typeof(D) E;                // E has type "double[4]" from D’s completed type
