// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
typedef int * Int_ptr_t;
typedef int Int_t;

void local_array_subscript_simple() {
  int tmp;
  int *p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  const int *q = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::span<const int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:29-[[@LINE-3]]:29}:", 10}"
  tmp = p[5];
  tmp = q[5];

  Int_ptr_t x = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:16}:"std::span<int> x"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:28-[[@LINE-3]]:28}:", 10}"
  Int_ptr_t y = new int;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:16}:"std::span<int> y"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 1}"
  Int_t * z = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<Int_t> z"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:", 10}"
  Int_t * w = new Int_t[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<Int_t> w"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:28-[[@LINE-3]]:28}:", 10}"

  tmp = x[5];
  tmp = y[5]; // y[5] will crash after being span
  tmp = z[5];
  tmp = w[5];
}

void local_array_subscript_auto() {
  int tmp;
  auto p = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"
  tmp = p[5];
}

void local_array_subscript_variable_extent() {
  int n = 10;
  int tmp;
  int *p = new int[n];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:", n}"
  // If the extent expression does not have a constant value, we cannot fill the extent for users...
  int *q = new int[n++];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
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
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", 10}"
  int *q = b;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:11}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:13}:", <# placeholder #>}"
  // No way to know if `n` is ever mutated since `int b[n];`, so no way to figure out the extent
  tmp = p[5];
  tmp = q[5];
}

void local_ptr_addrof_init() {
  int var;
  int * q = &var;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:", 1}"
  // This expression involves unsafe buffer accesses, which will crash
  // at runtime after applying the fix-it,
  var = q[5];
}

void decl_without_init() {
  int tmp;
  int * p;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:10}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  Int_ptr_t q;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int> q"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  tmp = p[5];
  tmp = q[5];
}

// Explicit casts are required in the following cases. No way to
// figure out span extent for them automatically.
void explict_cast() {
  int tmp;
  int * p = (int*) new int[10][10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:35-[[@LINE-3]]:35}:", <# placeholder #>}"
  tmp = p[5];

  int a;
  char * q = (char *)&a;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:13}:"std::span<char> q"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"
  tmp = (int) q[5];

  void * r = &a;
  char * s = (char *) r;
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:13}:"std::span<char> s"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", <# placeholder #>}"
  tmp = (int) s[5];
}

void null_init() {
#define NULL 0
  int tmp;
  int * my_null = 0;
  int * p = 0;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}
  int * g = NULL; // cannot handle fix-its involving macros for now
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * f = nullptr;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> f"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{^3}}

  // In case of value dependencies, we give up
  int * q = my_null;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> q"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:", <# placeholder #>}"
  int * r = my_null + 0;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> r"
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
  // CHECK-NOT: fix-it:"{{.*}}":[[@LINE-1]]
  *p = q[5];
}

void unsupported_fixit_overlapping_macro(int * x) {
  int tmp;
  // In the case below, a tentative fix-it replaces `MY_INT * p =` with `std::span<MY_INT> p `.
  // It overlaps with the use of the macro `MY_INT`.  The fix-it is
  // discarded then.
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
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> g"
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
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> r"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

  tmp = r[j] + r[k]; // both `j` and `k` are unsigned so they must be non-negative
  tmp = r[(unsigned int)-1]; // a cast-to-unsigned-expression is also non-negative
}
