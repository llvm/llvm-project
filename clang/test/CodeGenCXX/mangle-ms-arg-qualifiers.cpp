// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck -check-prefix=X64 %s

void foo(const unsigned int) {}
// CHECK: "?foo@@YAXI@Z"
// X64: "?foo@@YAXI@Z"

void foo(const double) {}
// CHECK: "?foo@@YAXN@Z"
// X64: "?foo@@YAXN@Z"

void bar(const volatile double) {}
// CHECK: "?bar@@YAXN@Z"
// X64: "?bar@@YAXN@Z"

void foo_pad(char * x) {}
// CHECK: "?foo_pad@@YAXPAD@Z"
// X64: "?foo_pad@@YAXPEAD@Z"

void foo_pbd(const char * x) {}
// CHECK: "?foo_pbd@@YAXPBD@Z"
// X64: "?foo_pbd@@YAXPEBD@Z"

void foo_pcd(volatile char * x) {}
// CHECK: "?foo_pcd@@YAXPCD@Z"
// X64: "?foo_pcd@@YAXPECD@Z"

void foo_qad(char * const x) {}
// CHECK: "?foo_qad@@YAXQAD@Z"
// X64: "?foo_qad@@YAXQEAD@Z"

void foo_rad(char * volatile x) {}
// CHECK: "?foo_rad@@YAXRAD@Z"
// X64: "?foo_rad@@YAXREAD@Z"

void foo_sad(char * const volatile x) {}
// CHECK: "?foo_sad@@YAXSAD@Z"
// X64: "?foo_sad@@YAXSEAD@Z"

void foo_piad(char * __restrict x) {}
// CHECK: "?foo_piad@@YAXPIAD@Z"
// X64: "?foo_piad@@YAXPEIAD@Z"

void foo_qiad(char * const __restrict x) {}
// CHECK: "?foo_qiad@@YAXQIAD@Z"
// X64: "?foo_qiad@@YAXQEIAD@Z"

void foo_riad(char * volatile __restrict x) {}
// CHECK: "?foo_riad@@YAXRIAD@Z"
// X64: "?foo_riad@@YAXREIAD@Z"

void foo_siad(char * const volatile __restrict x) {}
// CHECK: "?foo_siad@@YAXSIAD@Z"
// X64: "?foo_siad@@YAXSEIAD@Z"

void foo_papad(char ** x) {}
// CHECK: "?foo_papad@@YAXPAPAD@Z"
// X64: "?foo_papad@@YAXPEAPEAD@Z"

void foo_papbd(char const ** x) {}
// CHECK: "?foo_papbd@@YAXPAPBD@Z"
// X64: "?foo_papbd@@YAXPEAPEBD@Z"

void foo_papcd(char volatile ** x) {}
// CHECK: "?foo_papcd@@YAXPAPCD@Z"
// X64: "?foo_papcd@@YAXPEAPECD@Z"

void foo_pbqad(char * const* x) {}
// CHECK: "?foo_pbqad@@YAXPBQAD@Z"
// X64: "?foo_pbqad@@YAXPEBQEAD@Z"

void foo_pcrad(char * volatile* x) {}
// CHECK: "?foo_pcrad@@YAXPCRAD@Z"
// X64: "?foo_pcrad@@YAXPECREAD@Z"

void foo_qapad(char ** const x) {}
// CHECK: "?foo_qapad@@YAXQAPAD@Z"
// X64: "?foo_qapad@@YAXQEAPEAD@Z"

void foo_rapad(char ** volatile x) {}
// CHECK: "?foo_rapad@@YAXRAPAD@Z"
// X64: "?foo_rapad@@YAXREAPEAD@Z"

void foo_pbqbd(const char * const* x) {}
// CHECK: "?foo_pbqbd@@YAXPBQBD@Z"
// X64: "?foo_pbqbd@@YAXPEBQEBD@Z"

void foo_pbqcd(volatile char * const* x) {}
// CHECK: "?foo_pbqcd@@YAXPBQCD@Z"
// X64: "?foo_pbqcd@@YAXPEBQECD@Z"

void foo_pcrbd(const char * volatile* x) {}
// CHECK: "?foo_pcrbd@@YAXPCRBD@Z"
// X64: "?foo_pcrbd@@YAXPECREBD@Z"

void foo_pcrcd(volatile char * volatile* x) {}
// CHECK: "?foo_pcrcd@@YAXPCRCD@Z"
// X64: "?foo_pcrcd@@YAXPECRECD@Z"

void foo_aad(char &x) {}
// CHECK: "?foo_aad@@YAXAAD@Z"
// X64: "?foo_aad@@YAXAEAD@Z"

void foo_abd(const char &x) {}
// CHECK: "?foo_abd@@YAXABD@Z"
// X64: "?foo_abd@@YAXAEBD@Z"

void foo_aapad(char *&x) {}
// CHECK: "?foo_aapad@@YAXAAPAD@Z"
// X64: "?foo_aapad@@YAXAEAPEAD@Z"

void foo_aapbd(const char *&x) {}
// CHECK: "?foo_aapbd@@YAXAAPBD@Z"
// X64: "?foo_aapbd@@YAXAEAPEBD@Z"

void foo_abqad(char * const &x) {}
// CHECK: "?foo_abqad@@YAXABQAD@Z"
// X64: "?foo_abqad@@YAXAEBQEAD@Z"

void foo_abqbd(const char * const &x) {}
// CHECK: "?foo_abqbd@@YAXABQBD@Z"
// X64: "?foo_abqbd@@YAXAEBQEBD@Z"

void foo_aay144h(int (&x)[5][5]) {}
// CHECK: "?foo_aay144h@@YAXAAY144H@Z"
// X64: "?foo_aay144h@@YAXAEAY144H@Z"

void foo_aay144cbh(const int (&x)[5][5]) {}
// CHECK: "?foo_aay144cbh@@YAXAAY144$$CBH@Z"
// X64: "?foo_aay144cbh@@YAXAEAY144$$CBH@Z"

void foo_qay144h(int (&&x)[5][5]) {}
// CHECK: "?foo_qay144h@@YAX$$QAY144H@Z"
// X64: "?foo_qay144h@@YAX$$QEAY144H@Z"

void foo_qay144cbh(const int (&&x)[5][5]) {}
// CHECK: "?foo_qay144cbh@@YAX$$QAY144$$CBH@Z"
// X64: "?foo_qay144cbh@@YAX$$QEAY144$$CBH@Z"

void foo_p6ahxz(int x()) {}
// CHECK: "?foo_p6ahxz@@YAXP6AHXZ@Z"
// X64: "?foo_p6ahxz@@YAXP6AHXZ@Z"

void foo_a6ahxz(int (&x)()) {}
// CHECK: "?foo_a6ahxz@@YAXA6AHXZ@Z"
// X64: "?foo_a6ahxz@@YAXA6AHXZ@Z"

void foo_q6ahxz(int (&&x)()) {}
// CHECK: "?foo_q6ahxz@@YAX$$Q6AHXZ@Z"
// X64: "?foo_q6ahxz@@YAX$$Q6AHXZ@Z"

void foo_qay04h(int x[5][5]) {}
// CHECK: "?foo_qay04h@@YAXQAY04H@Z"
// X64: "?foo_qay04h@@YAXQEAY04H@Z"

void foo_qay04cbh(const int x[5][5]) {}
// CHECK: "?foo_qay04cbh@@YAXQAY04$$CBH@Z"
// X64: "?foo_qay04cbh@@YAXQEAY04$$CBH@Z"

typedef double Vector[3];

void foo(Vector*) {}
// CHECK: "?foo@@YAXPAY02N@Z"
// X64: "?foo@@YAXPEAY02N@Z"

void foo(Vector) {}
// CHECK: "?foo@@YAXQAN@Z"
// X64: "?foo@@YAXQEAN@Z"

void foo_const(const Vector) {}
// CHECK: "?foo_const@@YAXQBN@Z"
// X64: "?foo_const@@YAXQEBN@Z"

void foo_volatile(volatile Vector) {}
// CHECK: "?foo_volatile@@YAXQCN@Z"
// X64: "?foo_volatile@@YAXQECN@Z"

void foo(Vector*, const Vector, const double) {}
// CHECK: "?foo@@YAXPAY02NQBNN@Z"
// X64: "?foo@@YAXPEAY02NQEBNN@Z"

typedef void (*ConstFunPtr)(int *const d);
void foo_fnptrconst(ConstFunPtr f) {  }
// CHECK: "?foo_fnptrconst@@YAXP6AXQAH@Z@Z"
// X64:   "?foo_fnptrconst@@YAXP6AXQEAH@Z@Z"

typedef void (*ArrayFunPtr)(int d[1]);
void foo_fnptrarray(ArrayFunPtr f) {  }
// CHECK: "?foo_fnptrarray@@YAXP6AXQAH@Z@Z"
// X64:   "?foo_fnptrarray@@YAXP6AXQEAH@Z@Z"

void foo_fnptrbackref1(ArrayFunPtr f1, ArrayFunPtr f2) {  }
// CHECK: "?foo_fnptrbackref1@@YAXP6AXQAH@Z1@Z"
// X64:   "?foo_fnptrbackref1@@YAXP6AXQEAH@Z1@Z"

void foo_fnptrbackref2(ArrayFunPtr f1, ConstFunPtr f2) {  }
// CHECK: "?foo_fnptrbackref2@@YAXP6AXQAH@Z1@Z"
// X64:   "?foo_fnptrbackref2@@YAXP6AXQEAH@Z1@Z"

typedef void (*NormalFunPtr)(int *d);
void foo_fnptrbackref3(ArrayFunPtr f1, NormalFunPtr f2) {  }
// CHECK: "?foo_fnptrbackref3@@YAXP6AXQAH@Z1@Z"
// X64:   "?foo_fnptrbackref3@@YAXP6AXQEAH@Z1@Z"

void foo_fnptrbackref4(NormalFunPtr f1, ArrayFunPtr f2) {  }
// CHECK: "?foo_fnptrbackref4@@YAXP6AXPAH@Z1@Z"
// X64:   "?foo_fnptrbackref4@@YAXP6AXPEAH@Z1@Z"

ArrayFunPtr ret_fnptrarray() { return 0; }
// CHECK: "?ret_fnptrarray@@YAP6AXQAH@ZXZ"
// X64:   "?ret_fnptrarray@@YAP6AXQEAH@ZXZ"

// Test that we mangle the forward decl when we have a redeclaration with a
// slightly different type.
void mangle_fwd(char * const x);
void mangle_fwd(char * x) {}
// CHECK: "?mangle_fwd@@YAXQAD@Z"
// X64:   "?mangle_fwd@@YAXQEAD@Z"

void mangle_no_fwd(char * x) {}
// CHECK: "?mangle_no_fwd@@YAXPAD@Z"
// X64:   "?mangle_no_fwd@@YAXPEAD@Z"

// The first argument gets mangled as-if it were written "int *const"
// The second arg should not form a backref because it isn't qualified
void mangle_no_backref0(int[], int *) {}
// CHECK: "?mangle_no_backref0@@YAXQAHPAH@Z"
// X64:   "?mangle_no_backref0@@YAXQEAHPEAH@Z"

void mangle_no_backref1(int[], int *const) {}
// CHECK: "?mangle_no_backref1@@YAXQAHQAH@Z"
// X64:   "?mangle_no_backref1@@YAXQEAHQEAH@Z"

typedef void fun_type(void);
typedef void (*ptr_to_fun_type)(void);

// Pointer to function types don't backref with function types
void mangle_no_backref2(fun_type, ptr_to_fun_type) {}
// CHECK: "?mangle_no_backref2@@YAXP6AXXZP6AXXZ@Z"
// X64:   "?mangle_no_backref2@@YAXP6AXXZP6AXXZ@Z"

void mangle_yes_backref0(int[], int []) {}
// CHECK: "?mangle_yes_backref0@@YAXQAH0@Z"
// X64:   "?mangle_yes_backref0@@YAXQEAH0@Z"

void mangle_yes_backref1(int *const, int *const) {}
// CHECK: "?mangle_yes_backref1@@YAXQAH0@Z"
// X64:   "?mangle_yes_backref1@@YAXQEAH0@Z"

void mangle_yes_backref2(fun_type *const[], ptr_to_fun_type const[]) {}
// CHECK: "?mangle_yes_backref2@@YAXQBQ6AXXZ0@Z"
// X64:   "?mangle_yes_backref2@@YAXQEBQ6AXXZ0@Z"

void mangle_yes_backref3(ptr_to_fun_type *const, void (**const)(void)) {}
// CHECK: "?mangle_yes_backref3@@YAXQAP6AXXZ0@Z"
// X64:   "?mangle_yes_backref3@@YAXQEAP6AXXZ0@Z"

void mangle_yes_backref4(int *const __restrict, int *const __restrict) {}
// CHECK: "?mangle_yes_backref4@@YAXQIAH0@Z"
// X64:   "?mangle_yes_backref4@@YAXQEIAH0@Z"

struct S {};
void pr23325(const S[1], const S[]) {}
// CHECK: "?pr23325@@YAXQBUS@@0@Z"
// X64:   "?pr23325@@YAXQEBUS@@0@Z"

void vla_arg(int i, int a[][i]) {}
// CHECK: "?vla_arg@@YAXHQAY0A@H@Z"
// X64:   "?vla_arg@@YAXHQEAY0A@H@Z"
