// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fblocks -fsafe-buffer-usage-suggestions -verify %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fblocks -fsafe-buffer-usage-suggestions \
// RUN:            %s 2>&1 | FileCheck %s

// FIXME: what about possible diagnostic message non-determinism?

// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]:
void parmsNoFix(int *p, int *q) {
  int * a = new int[10];
  int * b = new int[10]; //expected-warning{{'b' is an unsafe pointer used for buffer access}} \
			   expected-note{{change type of 'b' to 'std::span' to preserve bounds information}}

  a = p;
  a = q;
  b[5] = 5; // expected-note{{used in buffer access here}}
}

// CHECK: fix-it:{{.*}}:{[[@LINE+2]]:21-[[@LINE+2]]:27}:"std::span<int> p"
// CHECK: fix-it:{{.*}}:{[[@LINE+14]]:2-[[@LINE+14]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void parmsSingleton(int *p) {return parmsSingleton(std::span<int>(p, <# size #>));}\n"
void parmsSingleton(int *p) { //expected-warning{{'p' is an unsafe pointer used for buffer access}} \
			        expected-note{{change type of 'p' to 'std::span' to preserve bounds information}}
  // CHECK: fix-it:{{.*}}:{[[@LINE+3]]:3-[[@LINE+3]]:8}:"std::span<int>"
  // CHECK: fix-it:{{.*}}:{[[@LINE+2]]:13-[[@LINE+2]]:13}:"{"
  // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:24-[[@LINE+1]]:24}:", 10}"
  int * a = new int[10];
  // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:3-[[@LINE+1]]:8}:"std::span<int>"
  int * b; //expected-warning{{'b' is an unsafe pointer used for buffer access}} \
	     expected-note{{change type of 'b' to 'std::span' to preserve bounds information, and change 'a' to 'std::span' to propagate bounds information between them}}

  b = a;
  b[5] = 5; // expected-note{{used in buffer access here}}
  p[5] = 5; // expected-note{{used in buffer access here}}
}


// Parameters other than `r` all will be fixed
// CHECK: fix-it:{{.*}}:{[[@LINE+15]]:24-[[@LINE+15]]:30}:"std::span<int> p"
// CHECK  fix-it:{{.*}}:{[[@LINE+14]]:32-[[@LINE+14]]:39}:"std::span<int *> q"
// CHECK: fix-it:{{.*}}:{[[@LINE+13]]:41-[[@LINE+13]]:48}:"std::span<int> a"
// CHECK: fix-it:{{.*}}:{[[@LINE+23]]:2-[[@LINE+23]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void * multiParmAllFix(int *p, int **q, int a[], int * r) {return multiParmAllFix(std::span<int>(p, <# size #>), std::span<int *>(q, <# size #>), std::span<int>(a, <# size #>), r);}\n"

// repeat 2 more times as each of the 3 fixing parameters generates the set of fix-its above.

// CHECK: fix-it:{{.*}}:{[[@LINE+8]]:24-[[@LINE+8]]:30}:"std::span<int> p"
// CHECK  fix-it:{{.*}}:{[[@LINE+7]]:32-[[@LINE+7]]:39}:"std::span<int *> q"
// CHECK: fix-it:{{.*}}:{[[@LINE+6]]:41-[[@LINE+6]]:48}:"std::span<int> a"
// CHECK: fix-it:{{.*}}:{[[@LINE+16]]:2-[[@LINE+16]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void * multiParmAllFix(int *p, int **q, int a[], int * r) {return multiParmAllFix(std::span<int>(p, <# size #>), std::span<int *>(q, <# size #>), std::span<int>(a, <# size #>), r);}\n"
// CHECK: fix-it:{{.*}}:{[[@LINE+4]]:24-[[@LINE+4]]:30}:"std::span<int> p"
// CHECK  fix-it:{{.*}}:{[[@LINE+3]]:32-[[@LINE+3]]:39}:"std::span<int *> q"
// CHECK: fix-it:{{.*}}:{[[@LINE+2]]:41-[[@LINE+2]]:48}:"std::span<int> a"
// CHECK: fix-it:{{.*}}:{[[@LINE+12]]:2-[[@LINE+12]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void * multiParmAllFix(int *p, int **q, int a[], int * r) {return multiParmAllFix(std::span<int>(p, <# size #>), std::span<int *>(q, <# size #>), std::span<int>(a, <# size #>), r);}\n"
void * multiParmAllFix(int *p, int **q, int a[], int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}   expected-warning{{'q' is an unsafe pointer used for buffer access}} \
   expected-warning{{'a' is an unsafe pointer used for buffer access}} \
   expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'a' to safe types to make function 'multiParmAllFix' bounds-safe}} \
   expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p' and 'a' to safe types to make function 'multiParmAllFix' bounds-safe}} \
   expected-note{{change type of 'a' to 'std::span' to preserve bounds information, and change 'p' and 'q' to safe types to make function 'multiParmAllFix' bounds-safe}}
  int tmp;

  tmp = p[5]; // expected-note{{used in buffer access here}}
  tmp = a[5]; // expected-note{{used in buffer access here}}
  if (++q) {} // expected-note{{used in pointer arithmetic here}}
  return r;
}

void * multiParmAllFix(int *p, int **q, int a[], int * r);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:58-[[@LINE-2]]:58}:";\nvoid * multiParmAllFix(std::span<int> p, std::span<int *> q, std::span<int> a, int * r)"

// Fixing local variables implicates fixing parameters
void  multiParmLocalAllFix(int *p, int * r) {
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]:
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]:
  int * x; // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]:
  int * z; // expected-warning{{'z' is an unsafe pointer used for buffer access}}
  int * y;

  x = p;
  y = x; // FIXME: we do not fix `y = x` here as the `.data()` fix-it is not generally correct
  // `x` needs to be fixed so does the pointer assigned to `x`, i.e.,`p`
  x[5] = 5; // expected-note{{used in buffer access here}}
  z = r;
  // `z` needs to be fixed so does the pointer assigned to `z`, i.e.,`r`
  z[5] = 5; // expected-note{{used in buffer access here}}
  // Since `p` and `r` are parameters need to be fixed together,
  // fixing `x` involves fixing all `p`, `r` and `z`. Similar for
  // fixing `z`.
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]:


// Fixing parameters implicates fixing local variables
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]:
void  multiParmLocalAllFix2(int *p, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
                                                  expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int * x = new int[10];
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]:
  int * z = new int[10];
  // CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]:
  int * y;

  p = x;
  y = x;    // FIXME: we do not fix `y = x` here as the `.data()` fix-it is not generally correct
  p[5] = 5; // expected-note{{used in buffer access here}}
  r = z;
  r[5] = 5; // expected-note{{used in buffer access here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]:


// No fix emitted for any of the parameter since parameter `r` cannot be fixed
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void noneParmFix(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
					         expected-warning{{'q' is an unsafe pointer used for buffer access}} \
					         expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int tmp = p[5]; // expected-note{{used in buffer access here}}
  tmp = q[5];     // expected-note{{used in buffer access here}}
  r++;            // expected-note{{used in pointer arithmetic here}}
  tmp = r[5];     // expected-note{{used in buffer access here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]

// To show what if the `r` in `noneParmFix` can be fixed:
void noneParmFix_control(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						         expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q' and 'r' to safe types to make function 'noneParmFix_control' bounds-safe}} \
					                 expected-warning{{'q' is an unsafe pointer used for buffer access}} \
						         expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p' and 'r' to safe types to make function 'noneParmFix_control' bounds-safe}} \
					                 expected-warning{{'r' is an unsafe pointer used for buffer access}} \
						         expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p' and 'q' to safe types to make function 'noneParmFix_control' bounds-safe}}
  int tmp = p[5]; // expected-note{{used in buffer access here}}
  tmp = q[5];     // expected-note{{used in buffer access here}}
  if (++r) {}     // expected-note{{used in pointer arithmetic here}}
  tmp = r[5];     // expected-note{{used in buffer access here}}
}


// No fix emitted for any of the parameter since local variable `l` cannot be fixed.
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void noneParmOrLocalFix(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						        expected-warning{{'q' is an unsafe pointer used for buffer access}} \
						        expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int tmp = p[5];  // expected-note{{used in buffer access here}}
  tmp = q[5];      // expected-note{{used in buffer access here}}
  tmp = r[5];      // expected-note{{used in buffer access here}}
  // `l` and `r` must be fixed together while all parameters must be fixed together as well:
  int * l; l = r;     // expected-warning{{'l' is an unsafe pointer used for buffer access}}

  l++;             // expected-note{{used in pointer arithmetic here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]

// To show what if the `l` can be fixed in `noneParmOrLocalFix`:
void noneParmOrLocalFix_control(int * p, int * q, int * r) {// \
  expected-warning{{'p' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q', 'r', and 'l' to safe types to make function 'noneParmOrLocalFix_control' bounds-safe}} \
  expected-warning{{'q' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p', 'r', and 'l' to safe types to make function 'noneParmOrLocalFix_control' bounds-safe}} \
  expected-warning{{'r' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p', 'q', and 'l' to safe types to make function 'noneParmOrLocalFix_control' bounds-safe}}
  int tmp = p[5];  // expected-note{{used in buffer access here}}
  tmp = q[5];      // expected-note{{used in buffer access here}}
  tmp = r[5];      // expected-note{{used in buffer access here}}
  int * l;         // expected-warning{{'l' is an unsafe pointer used for buffer access}} \
		      expected-note{{change type of 'l' to 'std::span' to preserve bounds information, and change 'p', 'q', and 'r' to safe types to make function 'noneParmOrLocalFix_control' bounds-safe}}
  l = r;
  if (++l){};         // expected-note{{used in pointer arithmetic here}}
}


// No fix emitted for any of the parameter since local variable `l` cannot be fixed.
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void noneParmOrLocalFix2(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						         expected-warning{{'q' is an unsafe pointer used for buffer access}} \
						         expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int tmp = p[5]; // expected-note{{used in buffer access here}}
  tmp = q[5];     // expected-note{{used in buffer access here}}
  tmp = r[5];     // expected-note{{used in buffer access here}}

  int * a; a = r;
  int * b; b = a;
  int * l; l = b;    // expected-warning{{'l' is an unsafe pointer used for buffer access}}

  // `a, b, l` and parameters must be fixed together but `l` can't be fixed:
  l++;               // expected-note{{used in pointer arithmetic here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]

// To show what if the `l` can be fixed in `noneParmOrLocalFix2`:
void noneParmOrLocalFix2_control(int * p, int * q, int * r) { // \
  expected-warning{{'p' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'q', 'r', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix2_control' bounds-safe}}                 \
  expected-warning{{'q' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p', 'r', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix2_control' bounds-safe}}                 \
  expected-warning{{'r' is an unsafe pointer used for buffer access}} \
  expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p', 'q', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix2_control' bounds-safe}}
  int tmp = p[5]; // expected-note{{used in buffer access here}}
  tmp = q[5];     // expected-note{{used in buffer access here}}
  tmp = r[5];     // expected-note{{used in buffer access here}}

  int * a; a = r;
  int * b; b = a;
  int * l;  // expected-warning{{'l' is an unsafe pointer used for buffer access}} \
	       expected-note{{change type of 'l' to 'std::span' to preserve bounds information, and change 'p', 'q', 'r', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix2_control' bounds-safe}}

  l = b;
  if(++l){} // expected-note{{used in pointer arithmetic here}}
}

// No fix emitted for any of the parameter since local variable `l` cannot be fixed
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void noneParmOrLocalFix3(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						         expected-warning{{'q' is an unsafe pointer used for buffer access}} \
						         expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int tmp = p[5];  // expected-note{{used in buffer access here}}
  tmp = q[5];      // expected-note{{used in buffer access here}}
  tmp = r[5];      // expected-note{{used in buffer access here}}

  int * a; a = r;
  int * b; b = a;
  int * l; l = b;     // expected-warning{{'l' is an unsafe pointer used for buffer access}}

  l++;             // expected-note{{used in pointer arithmetic here}}

  int * x; x = p; // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  tmp = x[5];  // expected-note{{used in buffer access here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]

void noneParmOrLocalFix3_control(int * p, int * q, int * r) { // \
     expected-warning{{'p' is an unsafe pointer used for buffer access}} \
     expected-note{{change type of 'p' to 'std::span' to preserve bounds information, and change 'x', 'q', 'r', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix3_control' bounds-safe}}            \
     expected-warning{{'q' is an unsafe pointer used for buffer access}} \
     expected-note{{change type of 'q' to 'std::span' to preserve bounds information, and change 'p', 'x', 'r', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix3_control' bounds-safe}}            \
     expected-warning{{'r' is an unsafe pointer used for buffer access}} \
     expected-note{{change type of 'r' to 'std::span' to preserve bounds information, and change 'p', 'x', 'q', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix3_control' bounds-safe}}
  int tmp = p[5];  // expected-note{{used in buffer access here}}
  tmp = q[5];      // expected-note{{used in buffer access here}}
  tmp = r[5];      // expected-note{{used in buffer access here}}

  int * a; a = r;
  int * b; b = a;
  int * l;         // expected-warning{{'l' is an unsafe pointer used for buffer access}}   \
		      expected-note{{change type of 'l' to 'std::span' to preserve bounds information, and change 'p', 'x', 'q', 'r', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix3_control' bounds-safe}}

  l = b;
  if (++l){};         // expected-note{{used in pointer arithmetic here}}

  int * x;            // expected-warning{{'x' is an unsafe pointer used for buffer access}} \
		         expected-note{{change type of 'x' to 'std::span' to preserve bounds information, and change 'p', 'q', 'r', 'l', 'b', and 'a' to safe types to make function 'noneParmOrLocalFix3_control' bounds-safe}}
  x = p;
  tmp = x[5];  // expected-note{{used in buffer access here}}
}


// No fix emitted for any of the parameter but some local variables will get fixed
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE+1]]
void noneParmSomeLocalFix(int * p, int * q, int * r) { // expected-warning{{'p' is an unsafe pointer used for buffer access}} \
						          expected-warning{{'q' is an unsafe pointer used for buffer access}} \
						          expected-warning{{'r' is an unsafe pointer used for buffer access}}
  int tmp = p[5];  // expected-note{{used in buffer access here}}
  tmp = q[5];      // expected-note{{used in buffer access here}}
  tmp = r[5];      // expected-note{{used in buffer access here}}

  int * a; a = r;
  int * b; b = a;
  int * l; l = b; // expected-warning{{'l' is an unsafe pointer used for buffer access}}

  l++; // expected-note{{used in pointer arithmetic here}}

  //`x` and `y` can be fixed
  int * x = new int[10];
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  // CHECK: fix-it:{{.*}}:{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:{{.*}}:{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:3-[[@LINE+1]]:8}:"std::span<int>"
  int * y;   // expected-warning{{'y' is an unsafe pointer used for buffer access}} \
	        expected-note{{change type of 'y' to 'std::span' to preserve bounds information, and change 'x' to 'std::span' to propagate bounds information between them}}
  y = x;
  tmp = y[5];  // expected-note{{used in buffer access here}}
}
// CHECK-NOT: fix-it:{{.*}}:{[[@LINE-1]]

// Test that other parameters of (lambdas and blocks) do not interfere
// the grouping of variables of the function.
// CHECK: fix-it:{{.*}}:{[[@LINE+3]]:30-[[@LINE+3]]:37}:"std::span<int> p"
// CHECK: fix-it:{{.*}}:{[[@LINE+2]]:39-[[@LINE+2]]:46}:"std::span<int> q"
// CHECK: fix-it:{{.*}}:{[[@LINE+20]]:2-[[@LINE+20]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void parmsFromLambdaAndBlock(int * p, int * q) {return parmsFromLambdaAndBlock(std::span<int>(p, <# size #>), std::span<int>(q, <# size #>));}\n"
void parmsFromLambdaAndBlock(int * p, int * q) {
  // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:3-[[@LINE+1]]:8}:"std::span<int>"
  int * a; // expected-warning{{'a' is an unsafe pointer used for buffer access}} \
	      expected-note{{change type of 'a' to 'std::span' to preserve bounds information, and change 'p', 'b', and 'q' to safe types to make function 'parmsFromLambdaAndBlock' bounds-safe}}
  // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:3-[[@LINE+1]]:8}:"std::span<int>"
  int * b; // expected-warning{{'b' is an unsafe pointer used for buffer access}} \
              expected-note{{change type of 'b' to 'std::span' to preserve bounds information, and change 'a', 'p', and 'q' to safe types to make function 'parmsFromLambdaAndBlock' bounds-safe}}
  auto Lamb = [](int * x) -> void { // expected-warning{{'x' is an unsafe pointer used for buffer access}}
    x[5] = 5;                       // expected-note{{used in buffer access here}}
  };

  void (^Blk)(int*) = ^(int * y) {  // expected-warning{{'y' is an unsafe pointer used for buffer access}}
    y[5] = 5;                       // expected-note{{used in buffer access here}}
  };

  a = p;
  b = q;
  a[5] = 5; // expected-note{{used in buffer access here}}
  b[5] = 5; // expected-note{{used in buffer access here}}
}
