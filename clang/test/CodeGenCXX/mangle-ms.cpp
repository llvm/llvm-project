// RUN: %clang_cc1 -no-opaque-pointers -fblocks -emit-llvm %s -o - -triple=i386-pc-win32 -std=c++98 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fblocks -emit-llvm %s -o - -triple=x86_64-pc-win32 -std=c++98| FileCheck -check-prefix X64 %s
// RUN: %clang_cc1 -no-opaque-pointers -fblocks -emit-llvm %s -o - -triple=aarch64-pc-win32 -std=c++98 -DARM | FileCheck -check-prefixes=X64,ARM %s

int a;
// CHECK-DAG: @"?a@@3HA"

extern "C++" {
static int __attribute__((used)) ignore_transparent_context;
// CHECK-DAG: @ignore_transparent_context
}

namespace N {
  int b;
// CHECK-DAG: @"?b@N@@3HA"

  namespace {
    int anonymous;
// CHECK-DAG: @"?anonymous@?A0x{{[^@]*}}@N@@3HA"
  }
}

static int c;
// CHECK-DAG: @c

int _c(void) {return N::anonymous + c;}
// CHECK-DAG: @"?_c@@YAHXZ"
// X64-DAG:   @"?_c@@YAHXZ"

const int &NeedsReferenceTemporary = 2;
// CHECK-DAG: @"?NeedsReferenceTemporary@@3ABHB" = dso_local constant i32* @"?$RT1@NeedsReferenceTemporary@@3ABHB"
// X64-DAG: @"?NeedsReferenceTemporary@@3AEBHEB" = dso_local constant i32* @"?$RT1@NeedsReferenceTemporary@@3AEBHEB"

class foo {
  static const short d;
// CHECK-DAG: @"?d@foo@@0FB"
protected:
  static volatile long e;
// CHECK-DAG: @"?e@foo@@1JC"
public:
  static const volatile char f;
// CHECK-DAG: @"?f@foo@@2DD"
  int operator+(int a);
  foo(){}
// CHECK-DAG: @"??0foo@@QAE@XZ"
// X64-DAG:   @"??0foo@@QEAA@XZ"

  ~foo(){}
// CHECK-DAG: @"??1foo@@QAE@XZ"
// X64-DAG:   @"??1foo@@QEAA@XZ

  foo(int i){}
// CHECK-DAG: @"??0foo@@QAE@H@Z"
// X64-DAG:   @"??0foo@@QEAA@H@Z"

  foo(char *q){}
// CHECK-DAG: @"??0foo@@QAE@PAD@Z"
// X64-DAG:   @"??0foo@@QEAA@PEAD@Z"

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
// CHECK-DAG: @"?bar@@YA?AVfoo@@XZ"
// X64-DAG:   @"?bar@@YA?AVfoo@@XZ"

int foo::operator+(int a) {
// CHECK-DAG: @"??Hfoo@@QAEHH@Z"
// X64-DAG:   @"??Hfoo@@QEAAHH@Z"

  foo::static_method();
// CHECK-DAG: @"?static_method@foo@@SAPAV1@XZ"
// X64-DAG:   @"?static_method@foo@@SAPEAV1@XZ"
  bar();
  return a;
}

const short foo::d = 0;
volatile long foo::e;
const volatile char foo::f = 'C';

int bar::g;
// CHECK-DAG: @"?g@bar@@2HA"

extern int * const h1 = &a;
// CHECK-DAG: @"?h1@@3QAHA"
extern const int * const h2 = &a;
// CHECK-DAG: @"?h2@@3QBHB"
extern int * const __restrict h3 = &a;
// CHECK-DAG: @"?h3@@3QIAHIA"
// X64-DAG: @"?h3@@3QEIAHEIA"

int i[10][20];
// CHECK-DAG: @"?i@@3PAY0BE@HA"

typedef int (*FunT)(int, int);
FunT FunArr[10][20];
// CHECK-DAG: @"?FunArr@@3PAY0BE@P6AHHH@ZA"
// X64-DAG: @"?FunArr@@3PAY0BE@P6AHHH@ZA"

int (__stdcall *j)(signed char, unsigned char);
// CHECK-DAG: @"?j@@3P6GHCE@ZA"

const char foo2::*m;
// CHECK-DAG: @"?m@@3PRfoo@@DR1@"
// X64-DAG:   @"?m@@3PERfoo@@DER1@"

const volatile char foo2::*k;
// CHECK-DAG: @"?k@@3PTfoo@@DT1@"
// X64-DAG:   @"?k@@3PETfoo@@DET1@"

int (foo2::*l)(int);
// CHECK-DAG: @"?l@@3P8foo@@AEHH@ZQ1@"

// Ensure typedef CV qualifiers are mangled correctly
typedef const int cInt;
typedef volatile int vInt;
typedef const volatile int cvInt;

extern cInt g_cInt = 1;
vInt g_vInt = 2;
cvInt g_cvInt = 3;

// CHECK-DAG: @"?g_cInt@@3HB"
// CHECK-DAG: @"?g_vInt@@3HC"
// CHECK-DAG: @"?g_cvInt@@3HD"

// Static functions are mangled, too.
// Also make sure calling conventions, arglists, and throw specs work.
static void __stdcall alpha(float a, double b) throw() {}
bool __fastcall beta(long long a, wchar_t b) throw(signed char, unsigned char) {
// CHECK-DAG: @"?beta@@YI_N_J_W@Z"
// X64-DAG:   @"?beta@@YA_N_J_W@Z"
  alpha(0.f, 0.0);
  return false;
}

// CHECK-DAG: @"?alpha@@YGXMN@Z"
// X64-DAG:   @"?alpha@@YAXMN@Z"

// Make sure tag-type mangling works.
void gamma(class foo, struct bar, union baz, enum quux) {}
// CHECK-DAG: @"?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"
// X64-DAG:   @"?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"

// Make sure pointer/reference-type mangling works.
void delta(int * const a, const long &) {}
// CHECK-DAG: @"?delta@@YAXQAHABJ@Z"
// X64-DAG:   @"?delta@@YAXQEAHAEBJ@Z"

// Array mangling.
void epsilon(int a[][10][20]) {}
// CHECK-DAG: @"?epsilon@@YAXQAY19BE@H@Z"
// X64-DAG:   @"?epsilon@@YAXQEAY19BE@H@Z"

void zeta(int (*)(int, int)) {}
// CHECK-DAG: @"?zeta@@YAXP6AHHH@Z@Z"
// X64-DAG:   @"?zeta@@YAXP6AHHH@Z@Z"

// Blocks mangling (Clang extension). A block should be mangled slightly
// differently from a similar function pointer.
void eta(int (^)(int, int)) {}
// CHECK-DAG: @"?eta@@YAXP_EAHHH@Z@Z"

typedef int theta_arg(int,int);
void theta(theta_arg^ block) {}
// CHECK-DAG: @"?theta@@YAXP_EAHHH@Z@Z"

void operator_new_delete() {
  char *ptr = new char;
// CHECK-DAG: @"??2@YAPAXI@Z"

  delete ptr;
// CHECK-DAG: @"??3@YAXPAX@Z"

  char *array = new char[42];
// CHECK-DAG: @"??_U@YAPAXI@Z"

  delete [] array;
// CHECK-DAG: @"??_V@YAXPAX@Z"
}

// PR13022
void (redundant_parens)();
void redundant_parens_use() { redundant_parens(); }
// CHECK-DAG: @"?redundant_parens@@YAXXZ"
// X64-DAG:   @"?redundant_parens@@YAXXZ"

// PR13047
typedef double RGB[3];
RGB color1;
// CHECK-DAG: @"?color1@@3PANA"
extern const RGB color2 = {};
// CHECK-DAG: @"?color2@@3QBNB"
extern RGB const color3[5] = {};
// CHECK-DAG: @"?color3@@3QAY02$$CBNA"
extern RGB const ((color4)[5]) = {};
// CHECK-DAG: @"?color4@@3QAY02$$CBNA"

struct B;
volatile int B::* volatile memptr1;
// X64-DAG: @"?memptr1@@3RESB@@HES1@"
volatile int B::* memptr2;
// X64-DAG: @"?memptr2@@3PESB@@HES1@"
int B::* volatile memptr3;
// X64-DAG: @"?memptr3@@3REQB@@HEQ1@"
typedef int (*fun)();
volatile fun B::* volatile funmemptr1;
// X64-DAG: @"?funmemptr1@@3RESB@@R6AHXZES1@"
volatile fun B::* funmemptr2;
// X64-DAG: @"?funmemptr2@@3PESB@@R6AHXZES1@"
fun B::* volatile funmemptr3;
// X64-DAG: @"?funmemptr3@@3REQB@@P6AHXZEQ1@"
void (B::* volatile memptrtofun1)();
// X64-DAG: @"?memptrtofun1@@3R8B@@EAAXXZEQ1@"
const void (B::* memptrtofun2)();
// X64-DAG: @"?memptrtofun2@@3P8B@@EAAXXZEQ1@"
volatile void (B::* memptrtofun3)();
// X64-DAG: @"?memptrtofun3@@3P8B@@EAAXXZEQ1@"
int (B::* volatile memptrtofun4)();
// X64-DAG: @"?memptrtofun4@@3R8B@@EAAHXZEQ1@"
volatile int (B::* memptrtofun5)();
// X64-DAG: @"?memptrtofun5@@3P8B@@EAA?CHXZEQ1@"
const int (B::* memptrtofun6)();
// X64-DAG: @"?memptrtofun6@@3P8B@@EAA?BHXZEQ1@"
fun (B::* volatile memptrtofun7)();
// X64-DAG: @"?memptrtofun7@@3R8B@@EAAP6AHXZXZEQ1@"
volatile fun (B::* memptrtofun8)();
// X64-DAG: @"?memptrtofun8@@3P8B@@EAAR6AHXZXZEQ1@"
const fun (B::* memptrtofun9)();
// X64-DAG: @"?memptrtofun9@@3P8B@@EAAQ6AHXZXZEQ1@"

// PR12603
enum E {};
// CHECK-DAG: "?fooE@@YA?AW4E@@XZ"
// X64-DAG:   "?fooE@@YA?AW4E@@XZ"
E fooE() { return E(); }

class X {};
// CHECK-DAG: "?fooX@@YA?AVX@@XZ"
// X64-DAG:   "?fooX@@YA?AVX@@XZ"
X fooX() { return X(); }

namespace PR13182 {
  extern char s0[];
  // CHECK-DAG: @"?s0@PR13182@@3PADA"
  extern char s1[42];
  // CHECK-DAG: @"?s1@PR13182@@3PADA"
  extern const char s2[];
  // CHECK-DAG: @"?s2@PR13182@@3QBDB"
  extern const char s3[42];
  // CHECK-DAG: @"?s3@PR13182@@3QBDB"
  extern volatile char s4[];
  // CHECK-DAG: @"?s4@PR13182@@3RCDC"
  extern const volatile char s5[];
  // CHECK-DAG: @"?s5@PR13182@@3SDDD"
  extern const char* const* s6;
  // CHECK-DAG: @"?s6@PR13182@@3PBQBDB"

  char foo() {
    return s0[0] + s1[0] + s2[0] + s3[0] + s4[0] + s5[0] + s6[0][0];
  }
}

extern "C" inline void extern_c_func() {
  static int local;
// CHECK-DAG: @"?local@?1??extern_c_func@@9@4HA"
// X64-DAG:   @"?local@?1??extern_c_func@@9@4HA"
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
  // CHECK: @"?static_variable_in_inline_function@?1??inline_function_with_local_type@@YAHXZ@4U<unnamed-type-static_variable_in_inline_function>@?1??1@YAHXZ@A"

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
  // CHECK: @"?static_variable_in_templated_inline_function@?1???$templated_inline_function_with_local_type@H@@YAHXZ@4U<unnamed-type-static_variable_in_templated_inline_function>@?1???$templated_inline_function_with_local_type@H@@YAHXZ@A"

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

void __vectorcall vector_func() { }
// CHECK-DAG: @"?vector_func@@YQXXZ"

template <void (*)(void)>
void fn_tmpl() {}

template void fn_tmpl<extern_c_func>();
// CHECK-DAG: @"??$fn_tmpl@$1?extern_c_func@@YAXXZ@@YAXXZ"

extern "C" void __attribute__((overloadable)) overloaded_fn() {}
// CHECK-DAG: @"?overloaded_fn@@$$J0YAXXZ"

extern "C" void overloaded_fn2() {}
// CHECK-DAG: @overloaded_fn2
//
extern "C" void __attribute__((overloadable)) overloaded_fn3();
extern "C" void overloaded_fn3() {}
// CHECK-DAG: @overloaded_fn3

namespace UnnamedType {
struct S {
  typedef struct {} *T1[1];
  typedef struct {} T2;
  typedef struct {} *T3, T4;
  using T5 = struct {};
  using T6 = struct {} *;
};
void f(S::T1) {}
void f(S::T2) {}
void f(S::T3) {}
void f(S::T4) {}
void f(S::T5) {}
void f(S::T6) {}
// CHECK-DAG: @"?f@UnnamedType@@YAXQAPAU<unnamed-type-T1>@S@1@@Z"
// CHECK-DAG: @"?f@UnnamedType@@YAXUT2@S@1@@Z"
// CHECK-DAG: @"?f@UnnamedType@@YAXPAUT4@S@1@@Z"
// CHECK-DAG: @"?f@UnnamedType@@YAXUT4@S@1@@Z"
// CHECK-DAG: @"?f@UnnamedType@@YAXUT5@S@1@@Z"
// CHECK-DAG: @"?f@UnnamedType@@YAXPAU<unnamed-type-T6>@S@1@@Z"

// X64-DAG: @"?f@UnnamedType@@YAXQEAPEAU<unnamed-type-T1>@S@1@@Z"
// X64-DAG: @"?f@UnnamedType@@YAXUT2@S@1@@Z"
// X64-DAG: @"?f@UnnamedType@@YAXPEAUT4@S@1@@Z"(%"struct.UnnamedType::S::T4"
// X64-DAG: @"?f@UnnamedType@@YAXUT4@S@1@@Z"
// X64-DAG: @"?f@UnnamedType@@YAXUT5@S@1@@Z"
// X64-DAG: @"?f@UnnamedType@@YAXPEAU<unnamed-type-T6>@S@1@@Z"
}

namespace PassObjectSize {
// NOTE: This mangling is subject to change.
// Reiterating from the comment in MicrosoftMangle, the scheme is pretend a
// parameter of type __clang::__pass_object_sizeN exists after each pass object
// size param P, where N is the Type of the pass_object_size attribute on P.
//
// e.g. we want to mangle:
//   void foo(void *const __attribute__((pass_object_size(0))));
// as if it were
//   namespace __clang { enum __pass_object_size0 : size_t {}; }
//   void foo(void *const, __clang::__pass_object_size0);
// where __clang is a top-level namespace.

// CHECK-DAG: define dso_local noundef i32 @"?foo@PassObjectSize@@YAHQAHW4__pass_object_size0@__clang@@@Z"
int foo(int *const i __attribute__((pass_object_size(0)))) { return 0; }
// CHECK-DAG: define dso_local noundef i32 @"?bar@PassObjectSize@@YAHQAHW4__pass_object_size1@__clang@@@Z"
int bar(int *const i __attribute__((pass_object_size(1)))) { return 0; }
// CHECK-DAG: define dso_local noundef i32 @"?qux@PassObjectSize@@YAHQAHW4__pass_object_size1@__clang@@0W4__pass_object_size0@3@@Z"
int qux(int *const i __attribute__((pass_object_size(1))), int *const j __attribute__((pass_object_size(0)))) { return 0; }
// CHECK-DAG: define dso_local noundef i32 @"?zot@PassObjectSize@@YAHQAHW4__pass_object_size1@__clang@@01@Z"
int zot(int *const i __attribute__((pass_object_size(1))), int *const j __attribute__((pass_object_size(1)))) { return 0; }
// CHECK-DAG: define dso_local noundef i32 @"?silly_word@PassObjectSize@@YAHQAHW4__pass_dynamic_object_size1@__clang@@@Z"
int silly_word(int *const i __attribute__((pass_dynamic_object_size(1)))) { return 0; }
}

namespace Atomic {
// CHECK-DAG: define dso_local void @"?f@Atomic@@YAXU?$_Atomic@H@__clang@@@Z"(
void f(_Atomic(int)) {}
}
namespace Complex {
// CHECK-DAG: define dso_local void @"?f@Complex@@YAXU?$_Complex@H@__clang@@@Z"(
void f(_Complex int) {}
}
#ifdef ARM
namespace Float16 {
// ARM-DAG: define dso_local void @"?f@Float16@@YAXU_Float16@__clang@@@Z"(
void f(_Float16) {}
}
#endif // ARM

namespace PR26029 {
template <class>
struct L {
  L() {}
};
template <class>
class H;
struct M : L<H<int *> > {};

template <class>
struct H {};

template <class GT>
void m_fn3() {
  (H<GT *>());
  M();
}

void runOnFunction() {
  L<H<int *> > b;
  m_fn3<int>();
}
// CHECK-DAG: call {{.*}} @"??0?$L@V?$H@PAH@PR26029@@@PR26029@@QAE@XZ"
}
