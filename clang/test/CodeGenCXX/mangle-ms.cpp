// RUN: %clang_cc1 -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=x86_64-pc-win32 -std=c++11| FileCheck -check-prefix X64 %s

int a;
// CHECK-DAG: @"\01?a@@3HA"

namespace N {
  int b;
// CHECK-DAG: @"\01?b@N@@3HA"

  namespace {
    int anonymous;
// CHECK-DAG: @"\01?anonymous@?A@N@@3HA"
  }
}

static int c;
// CHECK-DAG: @c

int _c(void) {return N::anonymous + c;}
// CHECK-DAG: @"\01?_c@@YAHXZ"
// X64-DAG:   @"\01?_c@@YAHXZ"

class foo {
  static const short d;
// CHECK-DAG: @"\01?d@foo@@0FB"
protected:
  static volatile long e;
// CHECK-DAG: @"\01?e@foo@@1JC"
public:
  static const volatile char f;
// CHECK-DAG: @"\01?f@foo@@2DD"
  int operator+(int a);
  foo(){}
// CHECK-DAG: @"\01??0foo@@QAE@XZ"
// X64-DAG:   @"\01??0foo@@QEAA@XZ"

  ~foo(){}
// CHECK-DAG: @"\01??1foo@@QAE@XZ"
// X64-DAG:   @"\01??1foo@@QEAA@XZ

  foo(int i){}
// CHECK-DAG: @"\01??0foo@@QAE@H@Z"
// X64-DAG:   @"\01??0foo@@QEAA@H@Z"

  foo(char *q){}
// CHECK-DAG: @"\01??0foo@@QAE@PAD@Z"
// X64-DAG:   @"\01??0foo@@QEAA@PEAD@Z"

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
// CHECK-DAG: @"\01?bar@@YA?AVfoo@@XZ"
// X64-DAG:   @"\01?bar@@YA?AVfoo@@XZ"

int foo::operator+(int a) {
// CHECK-DAG: @"\01??Hfoo@@QAEHH@Z"
// X64-DAG:   @"\01??Hfoo@@QEAAHH@Z"

  foo::static_method();
// CHECK-DAG: @"\01?static_method@foo@@SAPAV1@XZ"
// X64-DAG:   @"\01?static_method@foo@@SAPEAV1@XZ"
  bar();
  return a;
}

const short foo::d = 0;
volatile long foo::e;
const volatile char foo::f = 'C';

int bar::g;
// CHECK-DAG: @"\01?g@bar@@2HA"

extern int * const h1 = &a;
// CHECK-DAG: @"\01?h1@@3QAHA"
extern const int * const h2 = &a;
// CHECK-DAG: @"\01?h2@@3QBHB"

int i[10][20];
// CHECK-DAG: @"\01?i@@3PAY0BE@HA"

int (__stdcall *j)(signed char, unsigned char);
// CHECK-DAG: @"\01?j@@3P6GHCE@ZA"

const volatile char foo2::*k;
// CHECK-DAG: @"\01?k@@3PTfoo@@DT1@"
// X64-DAG:   @"\01?k@@3PETfoo@@DET1@"

int (foo2::*l)(int);
// CHECK-DAG: @"\01?l@@3P8foo@@AEHH@ZQ1@"

// Static functions are mangled, too.
// Also make sure calling conventions, arglists, and throw specs work.
static void __stdcall alpha(float a, double b) throw() {}
bool __fastcall beta(long long a, wchar_t b) throw(signed char, unsigned char) {
// CHECK-DAG: @"\01?beta@@YI_N_J_W@Z"
// X64-DAG:   @"\01?beta@@YA_N_J_W@Z"
  alpha(0.f, 0.0);
  return false;
}

// CHECK-DAG: @"\01?alpha@@YGXMN@Z"
// X64-DAG:   @"\01?alpha@@YAXMN@Z"

// Make sure tag-type mangling works.
void gamma(class foo, struct bar, union baz, enum quux) {}
// CHECK-DAG: @"\01?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"
// X64-DAG:   @"\01?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"

// Make sure pointer/reference-type mangling works.
void delta(int * const a, const long &) {}
// CHECK-DAG: @"\01?delta@@YAXQAHABJ@Z"
// X64-DAG:   @"\01?delta@@YAXQEAHAEBJ@Z"

// Array mangling.
void epsilon(int a[][10][20]) {}
// CHECK-DAG: @"\01?epsilon@@YAXQAY19BE@H@Z"
// X64-DAG:   @"\01?epsilon@@YAXQEAY19BE@H@Z"

void zeta(int (*)(int, int)) {}
// CHECK-DAG: @"\01?zeta@@YAXP6AHHH@Z@Z"
// X64-DAG:   @"\01?zeta@@YAXP6AHHH@Z@Z"

// Blocks mangling (Clang extension). A block should be mangled slightly
// differently from a similar function pointer.
void eta(int (^)(int, int)) {}
// CHECK-DAG: @"\01?eta@@YAXP_EAHHH@Z@Z"

typedef int theta_arg(int,int);
void theta(theta_arg^ block) {}
// CHECK-DAG: @"\01?theta@@YAXP_EAHHH@Z@Z"

void operator_new_delete() {
  char *ptr = new char;
// CHECK-DAG: @"\01??2@YAPAXI@Z"

  delete ptr;
// CHECK-DAG: @"\01??3@YAXPAX@Z"

  char *array = new char[42];
// CHECK-DAG: @"\01??_U@YAPAXI@Z"

  delete [] array;
// CHECK-DAG: @"\01??_V@YAXPAX@Z"
}

// PR13022
void (redundant_parens)();
void redundant_parens_use() { redundant_parens(); }
// CHECK-DAG: @"\01?redundant_parens@@YAXXZ"
// X64-DAG:   @"\01?redundant_parens@@YAXXZ"

// PR13047
typedef double RGB[3];
RGB color1;
// CHECK-DAG: @"\01?color1@@3PANA"
extern const RGB color2 = {};
// CHECK-DAG: @"\01?color2@@3QBNB"
extern RGB const color3[5] = {};
// CHECK-DAG: @"\01?color3@@3QAY02$$CBNA"
extern RGB const ((color4)[5]) = {};
// CHECK-DAG: @"\01?color4@@3QAY02$$CBNA"

struct B;
volatile int B::* volatile memptr1;
// X64-DAG: @"\01?memptr1@@3RESB@@HES1@"
volatile int B::* memptr2;
// X64-DAG: @"\01?memptr2@@3PESB@@HES1@"
int B::* volatile memptr3;
// X64-DAG: @"\01?memptr3@@3REQB@@HEQ1@"
typedef int (*fun)();
volatile fun B::* volatile funmemptr1;
// X64-DAG: @"\01?funmemptr1@@3RESB@@R6AHXZES1@"
volatile fun B::* funmemptr2;
// X64-DAG: @"\01?funmemptr2@@3PESB@@R6AHXZES1@"
fun B::* volatile funmemptr3;
// X64-DAG: @"\01?funmemptr3@@3REQB@@P6AHXZEQ1@"
void (B::* volatile memptrtofun1)();
// X64-DAG: @"\01?memptrtofun1@@3R8B@@EAAXXZEQ1@"
const void (B::* memptrtofun2)();
// X64-DAG: @"\01?memptrtofun2@@3P8B@@EAAXXZEQ1@"
volatile void (B::* memptrtofun3)();
// X64-DAG: @"\01?memptrtofun3@@3P8B@@EAAXXZEQ1@"
int (B::* volatile memptrtofun4)();
// X64-DAG: @"\01?memptrtofun4@@3R8B@@EAAHXZEQ1@"
volatile int (B::* memptrtofun5)();
// X64-DAG: @"\01?memptrtofun5@@3P8B@@EAA?CHXZEQ1@"
const int (B::* memptrtofun6)();
// X64-DAG: @"\01?memptrtofun6@@3P8B@@EAA?BHXZEQ1@"
fun (B::* volatile memptrtofun7)();
// X64-DAG: @"\01?memptrtofun7@@3R8B@@EAAP6AHXZXZEQ1@"
volatile fun (B::* memptrtofun8)();
// X64-DAG: @"\01?memptrtofun8@@3P8B@@EAAR6AHXZXZEQ1@"
const fun (B::* memptrtofun9)();
// X64-DAG: @"\01?memptrtofun9@@3P8B@@EAAQ6AHXZXZEQ1@"

// PR12603
enum E {};
// CHECK-DAG: "\01?fooE@@YA?AW4E@@XZ"
// X64-DAG:   "\01?fooE@@YA?AW4E@@XZ"
E fooE() { return E(); }

class X {};
// CHECK-DAG: "\01?fooX@@YA?AVX@@XZ"
// X64-DAG:   "\01?fooX@@YA?AVX@@XZ"
X fooX() { return X(); }

namespace PR13182 {
  extern char s0[];
  // CHECK-DAG: @"\01?s0@PR13182@@3PADA"
  extern char s1[42];
  // CHECK-DAG: @"\01?s1@PR13182@@3PADA"
  extern const char s2[];
  // CHECK-DAG: @"\01?s2@PR13182@@3QBDB"
  extern const char s3[42];
  // CHECK-DAG: @"\01?s3@PR13182@@3QBDB"
  extern volatile char s4[];
  // CHECK-DAG: @"\01?s4@PR13182@@3RCDC"
  extern const volatile char s5[];
  // CHECK-DAG: @"\01?s5@PR13182@@3SDDD"
  extern const char* const* s6;
  // CHECK-DAG: @"\01?s6@PR13182@@3PBQBDB"

  char foo() {
    return s0[0] + s1[0] + s2[0] + s3[0] + s4[0] + s5[0] + s6[0][0];
  }
}

extern "C" inline void extern_c_func() {
  static int local;
// CHECK-DAG: @"\01?local@?1??extern_c_func@@9@4HA"
// X64-DAG:   @"\01?local@?1??extern_c_func@@9@4HA"
}

void call_extern_c_func() {
  extern_c_func();
}

int main() { return 0; }
// CHECK-DAG: @main
// X64-DAG:   @main

int wmain() { return 0; }
// CHECK-DAG: @wmain
// X64-DAG:   @wmain

int WinMain() { return 0; }
// CHECK-DAG: @WinMain
// X64-DAG:   @WinMain

int wWinMain() { return 0; }
// CHECK-DAG: @wWinMain
// X64-DAG:   @wWinMain

int DllMain() { return 0; }
// CHECK-DAG: @DllMain
// X64-DAG:   @DllMain

inline int inline_function_with_local_type() {
  static struct {
    int a_field;
  } static_variable_in_inline_function = { 20 }, second_static = { 40 };
  // CHECK: @"\01?static_variable_in_inline_function@?1??inline_function_with_local_type@@YAHXZ@4U<unnamed-type-static_variable_in_inline_function>@?1??1@YAHXZ@A"

  return static_variable_in_inline_function.a_field + second_static.a_field;
}

int call_inline_function_with_local_type() {
  return inline_function_with_local_type();
}

template <typename T>
inline int templated_inline_function_with_local_type() {
  static struct {
    int a_field;
  } static_variable_in_templated_inline_function = { 20 },
    second_static = { 40 };
  // CHECK: @"\01?static_variable_in_templated_inline_function@?1???$templated_inline_function_with_local_type@H@@YAHXZ@4U<unnamed-type-static_variable_in_templated_inline_function>@?1???$templated_inline_function_with_local_type@H@@YAHXZ@A"

  return static_variable_in_templated_inline_function.a_field +
         second_static.a_field;
}

int call_templated_inline_function_with_local_type() {
  return templated_inline_function_with_local_type<int>();
}

// PR17371
struct OverloadedNewDelete {
  // __cdecl
  void *operator new(__SIZE_TYPE__);
  void *operator new[](__SIZE_TYPE__);
  void operator delete(void *);
  void operator delete[](void *);
  // __thiscall
  int operator+(int);
};

void *OverloadedNewDelete::operator new(__SIZE_TYPE__ s) { return 0; }
void *OverloadedNewDelete::operator new[](__SIZE_TYPE__ s) { return 0; }
void OverloadedNewDelete::operator delete(void *) { }
void OverloadedNewDelete::operator delete[](void *) { }
int OverloadedNewDelete::operator+(int x) { return x; };

// CHECK-DAG: ??2OverloadedNewDelete@@SAPAXI@Z
// CHECK-DAG: ??_UOverloadedNewDelete@@SAPAXI@Z
// CHECK-DAG: ??3OverloadedNewDelete@@SAXPAX@Z
// CHECK-DAG: ??_VOverloadedNewDelete@@SAXPAX@Z
// CHECK-DAG: ??HOverloadedNewDelete@@QAEHH@Z

// X64-DAG:   ??2OverloadedNewDelete@@SAPEAX_K@Z
// X64-DAG:   ??_UOverloadedNewDelete@@SAPEAX_K@Z
// X64-DAG:   ??3OverloadedNewDelete@@SAXPEAX@Z
// X64-DAG:   ??_VOverloadedNewDelete@@SAXPEAX@Z
// X64-DAG:   ??HOverloadedNewDelete@@QEAAHH@Z

// Indirecting the function type through a typedef will require a calling
// convention adjustment before building the method decl.

typedef void *__thiscall OperatorNewType(__SIZE_TYPE__);
typedef void __thiscall OperatorDeleteType(void *);

struct TypedefNewDelete {
  OperatorNewType operator new;
  OperatorNewType operator new[];
  OperatorDeleteType operator delete;
  OperatorDeleteType operator delete[];
};

void *TypedefNewDelete::operator new(__SIZE_TYPE__ s) { return 0; }
void *TypedefNewDelete::operator new[](__SIZE_TYPE__ s) { return 0; }
void TypedefNewDelete::operator delete(void *) { }
void TypedefNewDelete::operator delete[](void *) { }

// CHECK-DAG: ??2TypedefNewDelete@@SAPAXI@Z
// CHECK-DAG: ??_UTypedefNewDelete@@SAPAXI@Z
// CHECK-DAG: ??3TypedefNewDelete@@SAXPAX@Z
// CHECK-DAG: ??_VTypedefNewDelete@@SAXPAX@Z

namespace PR18022 {

struct { } a;
decltype(a) fun(decltype(a) x, decltype(a)) { return x; }
// CHECK-DAG: ?fun@PR18022@@YA?AU<unnamed-type-a>@1@U21@0@Z

}
