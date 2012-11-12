// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -fms-compatibility -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=x86_64-pc-win32 | FileCheck -check-prefix X64 %s

// CHECK: @"\01?a@@3HA"
// CHECK: @"\01?b@N@@3HA"
// CHECK: @c
// CHECK: @"\01?d@foo@@0FB"
// CHECK: @"\01?e@foo@@1JC"
// CHECK: @"\01?f@foo@@2DD"
// CHECK: @"\01?g@bar@@2HA"
// CHECK: @"\01?h1@@3QAHA"
// CHECK: @"\01?h2@@3QBHB"
// CHECK: @"\01?i@@3PAY0BE@HA"
// CHECK: @"\01?j@@3P6GHCE@ZA"
// CHECK: @"\01?k@@3PTfoo@@DA"
// CHECK: @"\01?l@@3P8foo@@AEHH@ZA"
// CHECK: @"\01?color1@@3PANA"
// CHECK: @"\01?color2@@3QBNB"

// FIXME: The following three tests currently fail, see http://llvm.org/PR13182
// Replace "CHECK-NOT" with "CHECK" when it is fixed.
// CHECK-NOT: @"\01?color3@@3QAY02$$CBNA"
// CHECK-NOT: @"\01?color4@@3QAY02$$CBNA"

int a;

namespace N { int b; }

static int c;
int _c(void) {return c;}
// CHECK: @"\01?_c@@YAHXZ"

class foo {
  static const short d;
protected:
  static volatile long e;
public:
  static const volatile char f;
  int operator+(int a);
  foo(){}
//CHECK: @"\01??0foo@@QAE@XZ"

  ~foo(){}
//CHECK: @"\01??1foo@@QAE@XZ"

  foo(int i){}
//CHECK: @"\01??0foo@@QAE@H@Z"

  foo(char *q){}
//CHECK: @"\01??0foo@@QAE@PAD@Z"

  static foo* static_method() { return 0; }

}f,s1(1),s2((char*)0);

typedef foo (foo2);

struct bar {
  static int g;
};

union baz {
  int a;
  char b;
  double c;
};

enum quux {
  qone,
  qtwo,
  qthree
};

foo bar() { return foo(); }
//CHECK: @"\01?bar@@YA?AVfoo@@XZ"

int foo::operator+(int a) {
//CHECK: @"\01??Hfoo@@QAEHH@Z"

  foo::static_method();
//CHECK: @"\01?static_method@foo@@SAPAV1@XZ"
  bar();
  return a;
}

const short foo::d = 0;
volatile long foo::e;
const volatile char foo::f = 'C';

int bar::g;

extern int * const h1 = &a;
extern const int * const h2 = &a;

int i[10][20];

int (__stdcall *j)(signed char, unsigned char);

const volatile char foo2::*k;

int (foo2::*l)(int);

// Static functions are mangled, too.
// Also make sure calling conventions, arglists, and throw specs work.
static void __stdcall alpha(float a, double b) throw() {}
bool __fastcall beta(long long a, wchar_t b) throw(signed char, unsigned char) {
// CHECK: @"\01?beta@@YI_N_J_W@Z"
  alpha(0.f, 0.0);
  return false;
}

// CHECK: @"\01?alpha@@YGXMN@Z"
// X64: @"\01?alpha@@YAXMN@Z"

// Make sure tag-type mangling works.
void gamma(class foo, struct bar, union baz, enum quux) {}
// CHECK: @"\01?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"

// Make sure pointer/reference-type mangling works.
void delta(int * const a, const long &) {}
// CHECK: @"\01?delta@@YAXQAHABJ@Z"

// Array mangling.
void epsilon(int a[][10][20]) {}
// CHECK: @"\01?epsilon@@YAXQAY19BE@H@Z"

void zeta(int (*)(int, int)) {}
// CHECK: @"\01?zeta@@YAXP6AHHH@Z@Z"

// Blocks mangling (Clang extension). A block should be mangled slightly
// differently from a similar function pointer.
void eta(int (^)(int, int)) {}
// CHECK: @"\01?eta@@YAXP_EAHHH@Z@Z"

typedef int theta_arg(int,int);
void theta(theta_arg^ block) {}
// CHECK: @"\01?theta@@YAXP_EAHHH@Z@Z"

void operator_new_delete() {
  char *ptr = new char;
// CHECK: @"\01??2@YAPAXI@Z"

  delete ptr;
// CHECK: @"\01??3@YAXPAX@Z"

  char *array = new char[42];
// CHECK: @"\01??_U@YAPAXI@Z"

  delete [] array;
// CHECK: @"\01??_V@YAXPAX@Z"
}

// PR13022
void (redundant_parens)();
void redundant_parens_use() { redundant_parens(); }
// CHECK: @"\01?redundant_parens@@YAXXZ"

// PR13047
typedef double RGB[3];
RGB color1;
extern const RGB color2 = {};
extern RGB const color3[5] = {};
extern RGB const ((color4)[5]) = {};

// PR12603
enum E {};
// CHECK: "\01?fooE@@YA?AW4E@@XZ"
E fooE() { return E(); }

class X {};
// CHECK: "\01?fooX@@YA?AVX@@XZ"
X fooX() { return X(); }

namespace PR13182 {
  extern char s0[];
  // CHECK: @"\01?s0@PR13182@@3PADA"
  extern char s1[42];
  // CHECK: @"\01?s1@PR13182@@3PADA"
  extern const char s2[];
  // CHECK: @"\01?s2@PR13182@@3QBDB"
  extern const char s3[42];
  // CHECK: @"\01?s3@PR13182@@3QBDB"
  extern volatile char s4[];
  // CHECK: @"\01?s4@PR13182@@3RCDC"
  extern const volatile char s5[];
  // CHECK: @"\01?s5@PR13182@@3SDDD"
  extern const char* const* s6;
  // CHECK: @"\01?s6@PR13182@@3PBQBDB"

  char foo() {
    return s0[0] + s1[0] + s2[0] + s3[0] + s4[0] + s5[0] + s6[0][0];
  }
}
