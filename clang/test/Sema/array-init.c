// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -std=gnu99 -fsyntax-only -Wgnu -Wc11-extensions -verify %s
// REQUIRES: LP64

extern int foof() = 1; // expected-error{{illegal initializer (only variables can be initialized)}}

static int x, y, z;

static int ary[] = { x, y, z }; // expected-error{{initializer element is not a compile-time constant}}
int ary2[] = { x, y, z }; // expected-error{{initializer element is not a compile-time constant}}

extern int fileScopeExtern[3] = { 1, 3, 5 }; // expected-warning{{'extern' variable has an initializer}}

static long ary3[] = { 1, "abc", 3, 4 }; // expected-warning{{incompatible pointer to integer conversion initializing 'long' with an expression of type 'char [4]'}}

void func() {
  int x = 1;

  typedef int TInt = 1; // expected-error{{illegal initializer (only variables can be initialized)}}

  int xComputeSize[] = { 1, 3, 5 };

  int x3[x] = { 1, 2 }; // expected-error{{variable-sized object may not be initialized}}

  int x4 = { 1, 2 }; // expected-warning{{excess elements in scalar initializer}}

  int y[4][3] = { 
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
  };

  int y2[4][3] = {
    1, 3, 5, 2, 4, 6, 3, 5, 7
  };

  int y3[4][3] = {  
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 },
    { 5 }, // expected-warning{{excess elements in array initializer}}
  };

  struct threeElements {
    int a,b,c;
  } z = { 1 };

  struct threeElements *p = 7; // expected-warning{{incompatible integer to pointer conversion initializing 'struct threeElements *' with an expression of type 'int'}}
  
  extern int blockScopeExtern[3] = { 1, 3, 5 }; // expected-error{{'extern' variable cannot have an initializer}}
  
  static long x2[3] = { 1.0,
                        "abc", // expected-warning{{incompatible pointer to integer conversion initializing 'long' with an expression of type 'char [4]'}}
                         5.8 }; // expected-warning {{implicit conversion from 'double' to 'long' changes value from 5.8 to 5}}
}

void test() {
  int y1[3] = { 
    { 1, 2, 3 } // expected-warning{{excess elements in scalar initializer}}
  };
  int y3[4][3] = {  
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 },
    {  }, // expected-warning{{use of GNU empty initializer extension}} expected-warning{{excess elements in array initializer}}
  };
  int y4[4][3] = {  
    { 1, 3, 5, 2 }, // expected-warning{{excess elements in array initializer}}
    { 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 },
  };
}

void allLegalAndSynonymous() {
  short q[4][3][2] = {
    { 1 },
    { 2, 3 },
    { 4, 5, 6 }
  };
  short q2[4][3][2] = {
    { 1, 0, 0, 0, 0, 0 },
    { 2, 3, 0, 0, 0, 0 },
    { 4, 5, 6 }
  };
  short q3[4][3][2] = {
    { 
      { 1 },
    },
    { 
      { 2, 3 },
    },
    { 
      { 4, 5 },
      { 6 },
    },
  };
}

void legal() {
  short q[][3][2] = {
    { 1 },
    { 2, 3 },
    { 4, 5, 6 }
  };
  int q_sizecheck[(sizeof(q) / sizeof(short [3][2])) == 3? 1 : -1];
}

unsigned char asso_values[] = { 34 };
int legal2() { 
  return asso_values[0]; 
}

void illegal() {
  short q2[4][][2] = { // expected-error{{array has incomplete element type 'short [][2]'}}
    { 1, 0, 0, 0, 0, 0 },
    { 2, 3, 0, 0, 0, 0 },
    { 4, 5, 6 }
  };
  short q3[4][3][] = { // expected-error{{array has incomplete element type 'short []'}}
    { 
      { 1 },
    },
    { 
      { 2, 3 },
    },
    { 
      { 4, 5 },
      { 6 },
    },
  };
  int a[][] = { 1, 2 }; // expected-error{{array has incomplete element type 'int []'}}
}

typedef int AryT[];

void testTypedef()
{
  AryT a = { 1, 2 }, b = { 3, 4, 5 };
  int a_sizecheck[(sizeof(a) / sizeof(int)) == 2? 1 : -1];
  int b_sizecheck[(sizeof(b) / sizeof(int)) == 3? 1 : -1];
}

static char const xx[] = "test";
int xx_sizecheck[(sizeof(xx) / sizeof(char)) == 5? 1 : -1];
static char const yy[5] = "test";
static char const zz[3] = "test"; // expected-warning{{initializer-string for char array is too long}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexcess-initializers"
static char const zz_quiet[3] = "test";
#pragma clang diagnostic pop

void charArrays() {
  static char const test[] = "test";
  int test_sizecheck[(sizeof(test) / sizeof(char)) == 5? 1 : -1];
  static char const test2[] = { "weird stuff" };
  static char const test3[] = { "test", "excess stuff" }; // expected-warning{{excess elements in char array initializer}}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexcess-initializers"
  static char const test3_quiet[] = {"test", "excess stuff"};
#pragma clang diagnostic pop

  char* cp[] = { "Hello" };

  char c[] = { "Hello" };
  int l[sizeof(c) == 6 ? 1 : -1];
  
  int i[] = { "Hello "}; // expected-warning{{incompatible pointer to integer conversion initializing 'int' with an expression of type 'char [7]'}}
  char c2[] = { "Hello", "Good bye" }; //expected-warning{{excess elements in char array initializer}}

  int i2[1] = { "Hello" }; //expected-warning{{incompatible pointer to integer conversion initializing 'int' with an expression of type 'char [6]'}}
  char c3[5] = { "Hello" };
  char c4[4] = { "Hello" }; //expected-warning{{initializer-string for char array is too long}}

  int i3[] = {}; //expected-warning{{zero size arrays are an extension}} expected-warning{{use of GNU empty initializer extension}}
}

void variableArrayInit() {
  int a = 4;
  char strlit[a] = "foo"; //expected-error{{variable-sized object may not be initialized}}
  int b[a] = { 1, 2, 4 }; //expected-error{{variable-sized object may not be initialized}}
}

// Pure array tests
float r1[10] = {{7}}; //expected-warning{{braces around scalar initializer}}
float r2[] = {{8}}; //expected-warning{{braces around scalar initializer}}
char r3[][5] = {1,2,3,4,5,6};
int r3_sizecheck[(sizeof(r3) / sizeof(char[5])) == 2? 1 : -1];
char r3_2[sizeof r3 == 10 ? 1 : -1];
float r4[1][2] = {1,{2},3,4}; //expected-warning{{braces around scalar initializer}} expected-warning{{excess elements in array initializer}}
char r5[][5] = {"aa", "bbb", "ccccc"};
char r6[sizeof r5 == 15 ? 1 : -1];
const char r7[] = "zxcv";
char r8[5] = "5char";
char r9[5] = "6chars"; //expected-warning{{initializer-string for char array is too long}}
unsigned char r10[] = __extension__ (_Generic(0, int: (__extension__ "foo" )));
int r11[0] = {}; //expected-warning{{zero size arrays are an extension}} expected-warning{{use of GNU empty initializer extension}}

// Some struct tests
void autoStructTest() {
struct s1 {char a; char b;} t1;
struct s2 {struct s1 c;} t2 = { t1 };
// The following is a less than great diagnostic (though it's on par with EDG).
struct s1 t3[] = {t1, t1, "abc", 0}; //expected-warning{{incompatible pointer to integer conversion initializing 'char' with an expression of type 'char [4]'}}
int t4[sizeof t3 == 6 ? 1 : -1];
}
struct foo { int z; } w;
int bar (void) { 
  struct foo z = { w }; //expected-error{{initializing 'int' with an expression of incompatible type 'struct foo'}}
  return z.z; 
} 
struct s3 {void (*a)(void);} t5 = {autoStructTest};
struct {int a; int b[];} t6 = {1, {1, 2, 3}}; // expected-warning{{flexible array initialization is a GNU extension}} \
// expected-note{{initialized flexible array member 'b' is here}}
union {char a; int b;} t7[] = {1, 2, 3};
int t8[sizeof t7 == (3*sizeof(int)) ? 1 : -1];

struct bittest{int : 31, a, :21, :12, b;};
struct bittest bittestvar = {1, 2, 3, 4}; //expected-warning{{excess elements in struct initializer}}

// Not completely sure what should happen here...
int u1 = {}; //expected-warning{{use of GNU empty initializer extension}} expected-error{{scalar initializer cannot be empty}}
int u2 = {{3}}; //expected-warning{{too many braces around scalar initializer}}

// PR2362
void varArray() {
  int c[][x] = { 0 }; //expected-error{{variable-sized object may not be initialized}}
}

// PR2151
void emptyInit() {struct {} x[] = {6};} //expected-warning{{empty struct is a GNU extension}} \
// expected-error{{initializer for aggregate with no elements}}

void noNamedInit() {
  struct {int:5;} x[] = {6}; //expected-error{{initializer for aggregate with no elements}} \
// expected-warning {{struct without named members is a GNU extension}}
}
struct {int a; int:5;} noNamedImplicit[] = {1,2,3};
int noNamedImplicitCheck[sizeof(noNamedImplicit) == 3 * sizeof(*noNamedImplicit) ? 1 : -1];


// ptrs are constant
struct soft_segment_descriptor {
  long ssd_base;
};
static int dblfault_tss;

union uniao { int ola; } xpto[1];

struct soft_segment_descriptor gdt_segs[] = {
  {(long) &dblfault_tss},
  { (long)xpto},
};

static void sppp_ipv6cp_up();
const struct {} ipcp = { sppp_ipv6cp_up }; //expected-warning{{empty struct is a GNU extension}} \
// expected-warning{{excess elements in struct initializer}}

struct _Matrix { union { float m[4][4]; }; }; //expected-warning{{anonymous unions are a C11 extension}}
typedef struct _Matrix Matrix;
void test_matrix() {
  const Matrix mat1 = {
    { { 1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f } }
  };

  const Matrix mat2 = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f 
  };
}

char badchararray[1] = { badchararray[0], "asdf" }; // expected-warning {{excess elements in array initializer}} expected-error {{initializer element is not a compile-time constant}}

// Test the GNU extension for initializing an array from an array
// compound literal. PR9261.
typedef int int5[5];
int a1[5] = (int[]){1, 2, 3, 4, 5}; // expected-warning{{initialization of an array of type 'int [5]' from a compound literal of type 'int [5]' is a GNU extension}}
int a2[5] = (int[5]){1, 2, 3, 4, 5}; // expected-warning{{initialization of an array of type 'int [5]' from a compound literal of type 'int [5]' is a GNU extension}}
int a3[] = ((int[]){1, 2, 3, 4, 5}); // expected-warning{{initialization of an array of type 'int []' from a compound literal of type 'int [5]' is a GNU extension}}
int a4[] = (int[5]){1, 2, 3, 4, 5}; // expected-warning{{initialization of an array of type 'int []' from a compound literal of type 'int [5]' is a GNU extension}}
int a5[] = (int5){1, 2, 3, 4, 5}; // expected-warning{{initialization of an array of type 'int []' from a compound literal of type 'int5' (aka 'int [5]') is a GNU extension}}

int a6[5] = (int[]){1, 2, 3}; // expected-error{{cannot initialize array of type 'int [5]' with array of type 'int [3]'}}

int nonconst_value();
int a7[5] = (int[5]){ 1,
                      2,
                      3,
                      4,
                      nonconst_value() // expected-error{{initializer element is not a compile-time constant}}
};

// <rdar://problem/10636946>
__attribute__((weak)) const unsigned int test10_bound = 10;
char test10_global[test10_bound]; // expected-error {{variable length array declaration not allowed at file scope}}
void test10() {
  char test10_local[test10_bound] = "help"; // expected-error {{variable-sized object may not be initialized}}
}
