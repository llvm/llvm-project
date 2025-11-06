// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
typedef int * Int_ptr_t;
typedef int Int_t;

void local_array_subscript_simple() {
  int tmp;
  int *p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  const int *q = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:29-[[@LINE-3]]:29}:", 10}"
  tmp = p[5];
  tmp = q[5];

  // We do not fix the following declaration. Because if the
  // definition of `Int_ptr_t` gets changed, the fixed code becomes
  // incorrect and may NOT be noticed.
  // FIXME: Fix with std::span<std::remove_pointer_t<Int_ptr_t>>?
  Int_ptr_t x = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  Int_t * z = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:10}:"std::span<Int_t>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:", 10}"
  Int_t * w = new Int_t[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:10}:"std::span<Int_t>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:28-[[@LINE-3]]:28}:", 10}"

  tmp = x[5];
  tmp = z[5];
  tmp = w[5];
}

void local_array_subscript_auto() {
  int tmp;
  // We do not fix the following declaration because
  // that'd cause us to hardcode the element type.
  // FIXME: Can we use the C++17 class template argument deduction
  // to avoid spelling out the element type?
  auto p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  tmp = p[5];
}

void local_variable_qualifiers_specifiers() {
  int a[10];
  const int * p = a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const>"
  const int * const q = a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const>"
  int tmp;
  tmp = p[5];
  tmp = q[5];

  [[deprecated]] const int * x = a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:18-[[@LINE-1]]:29}:"std::span<int const>"
  const int * y [[deprecated]];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const>"
  tmp = x[5];
  tmp = y[5];
}

void local_variable_unsupported_specifiers() {
  int a[10];
  const int * p [[deprecated]] = a; //  not supported because the attribute overlaps the source range of the declaration
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

  static const int * q = a; //  storage specifier not supported yet
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

  extern int * x; //  storage specifier not supported yet
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

  constexpr int * y = 0; //  `constexpr` specifier not supported yet
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

  int tmp;

  tmp = p[5];
  tmp = q[5];
  tmp = x[5];
  tmp = y[5];
}

void local_array_subscript_variable_extent() {
  int n = 10;
  int tmp;
  int *p = new int[n];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", n}"
  // If the extent expression does not have a constant value, we cannot fill the extent for users...
  int *q = new int[n++];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"
  tmp = p[5];
  tmp = q[5];
}


void local_ptr_to_array() {
  int tmp;
  int n = 10;
  int a[10];
  int b[n];  // If the extent expression does not have a constant value, we cannot fill the extent for users...
  int *p = a;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  int *q = b;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // No way to know if `n` is ever mutated since `int b[n];`, so no way to figure out the extent
  tmp = p[5];
  tmp = q[5];
}

void local_ptr_addrof_init() {
  int var;
  int * q = &var;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:", 1}"
  // This expression involves unsafe buffer accesses, which will crash
  // at runtime after applying the fix-it,
  var = q[5];
}

void decl_without_init() {
  int tmp;
  int * p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  Int_t * q;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:10}:"std::span<Int_t>"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  tmp = p[5];
  tmp = q[5];
}

// Explicit casts are required in the following cases. No way to
// figure out span extent for them automatically.
void explict_cast() {
  int tmp;
  int * p = (int*) new int[10][10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:35-[[@LINE-3]]:35}:", <# placeholder #>}"
  tmp = p[5];

  int a;
  char * q = (char *)&a;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:9}:"std::span<char>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"
  tmp = (int) q[5];

  void * r = &a;
  char * s = (char *) r;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:9}:"std::span<char>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"
  tmp = (int) s[5];
}

void null_init() {
#define NULL 0
  int tmp;
  int * my_null = 0;
  int * p = 0;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  int * g = NULL; // cannot handle fix-its involving macros for now
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * f = nullptr;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}

  // In case of value dependencies, we give up
  int * q = my_null;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:", <# placeholder #>}"
  int * r = my_null + 0;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"

  tmp = p[5]; // `p[5]` will cause crash after `p` being transformed to be a `std::span`
  tmp = q[5]; // Similar for the rests.
  tmp = r[5];
  tmp = g[5];
  tmp = f[5];
#undef NULL
}


void unsupported_multi_decl(int * x) {
  int * p = x, * q = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  *p = q[5];
}

void macroVariableIdentifier() {
#define MY_NAME p
#define MY_NAME_ARG(x) q

  // Although fix-its include macros, the macros do not overlap with
  // the bounds of the source range of these fix-its. So these fix-its
  // are valid.

  int * MY_NAME = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:19-[[@LINE-2]]:19}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:30-[[@LINE-3]]:30}:", 10}"
  int * MY_NAME_ARG( 'x' ) = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:30-[[@LINE-2]]:30}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:41-[[@LINE-3]]:41}:", 10}"
  p[5] = 5;
  q[5] = 5;
#undef MY_NAME
#undef MY_NAME_ARG
}

void unsupported_fixit_overlapping_macro(int * x) {
  int tmp;
  // In the case below, a tentative fix-it replaces `MY_INT * p =` with `std::span<MY_INT> p `.
  // The bounds of the source range of the fix-it overlap with the use of the macro
  // `MY_INT`.  The fix-it is discarded then.

  // FIXME: we do not have to discard a fix-it if its begin location
  // overlaps with the begin location of a macro. Similar for end
  // locations.

#define MY_INT int
  MY_INT * p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  tmp = p[5];

#define MY_VAR(name) int * name
  MY_VAR(q) = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  tmp = q[5];

  // In cases where fix-its do not change the original code where
  // macros are used, those fix-its will be emitted.  For example,
  // fixits are inserted before and after `new MY_INT[MY_TEN]` below.
#define MY_TEN 10
  int * g = new MY_INT[MY_TEN];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:31-[[@LINE-3]]:31}:", MY_TEN}"
  tmp = g[5];

#undef MY_INT
#undef MY_VAR
#undef MY_TEN
}

void unsupported_subscript_negative(int i, unsigned j, unsigned long k) {
  int tmp;
  int * p = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]

  tmp = p[-1]; // If `p` is made a span, this `[]` operation is wrong,
         // so no fix-it emitted.

  int * q = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]

  tmp = q[5];
  tmp = q[i];  // If `q` is made a span, this `[]` operation may be
         // wrong as we do not know if `i` is non-negative, so
         // no fix-it emitted.

  int * r = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

  tmp = r[j] + r[k]; // both `j` and `k` are unsigned so they must be non-negative
  tmp = r[(unsigned int)-1]; // a cast-to-unsigned-expression is also non-negative
}

#define DEFINE_PTR(X) int* ptr = (X);

void all_vars_in_macro() {
  int* local;
  DEFINE_PTR(local)
  ptr[1] = 0;
}

void few_vars_in_macro() {
  int* local;
  DEFINE_PTR(local)
  ptr[1] = 0;
  int tmp;
  ptr[2] = 30;
  int * p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  tmp = p[5];
  int val = *p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:14}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"[0]"
  val = *p + 30;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:10}:""
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:11-[[@LINE-2]]:11}:"[0]"
}
