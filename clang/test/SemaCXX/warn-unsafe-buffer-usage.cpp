// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fblocks -include %s -verify %s
#ifndef INCLUDED
#define INCLUDED
#pragma clang system_header

// no spanification warnings for system headers
void foo(...);  // let arguments of `foo` to hold testing expressions
void testAsSystemHeader(char *p) {
  ++p;

  auto ap1 = p;
  auto ap2 = &p;

  foo(p[1],
      ap1[1],
      ap2[2][3]);
}

#else

void testIncrement(char *p) {
  ++p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p++; // expected-warning{{unchecked operation on raw buffer in expression}}
  --p; // expected-warning{{unchecked operation on raw buffer in expression}}
  p--; // expected-warning{{unchecked operation on raw buffer in expression}}
}

void * voidPtrCall(void);
char * charPtrCall(void);

void testArraySubscripts(int *p, int **pp) {
  foo(p[1],             // expected-warning{{unchecked operation on raw buffer in expression}}
      pp[1][1],         // expected-warning2{{unchecked operation on raw buffer in expression}}
      1[1[pp]],         // expected-warning2{{unchecked operation on raw buffer in expression}}
      1[pp][1]          // expected-warning2{{unchecked operation on raw buffer in expression}}
      );

  if (p[3]) {           // expected-warning{{unchecked operation on raw buffer in expression}}
    void * q = p;

    foo(((int*)q)[10]); // expected-warning{{unchecked operation on raw buffer in expression}}
  }

  foo(((int*)voidPtrCall())[3], // expected-warning{{unchecked operation on raw buffer in expression}}
      3[(int*)voidPtrCall()],   // expected-warning{{unchecked operation on raw buffer in expression}}
      charPtrCall()[3],         // expected-warning{{unchecked operation on raw buffer in expression}}
      3[charPtrCall()]          // expected-warning{{unchecked operation on raw buffer in expression}}
      );

  int a[10], b[10][10];

  foo(a[1], 1[a], // expected-warning2{{unchecked operation on raw buffer in expression}}
  b[3][4],  // expected-warning2{{unchecked operation on raw buffer in expression}}
  4[b][3],  // expected-warning2{{unchecked operation on raw buffer in expression}}
  4[3[b]]); // expected-warning2{{unchecked operation on raw buffer in expression}}

  // Not to warn when index is zero
  foo(p[0], pp[0][0], 0[0[pp]], 0[pp][0],
      ((int*)voidPtrCall())[0],
      0[(int*)voidPtrCall()],
      charPtrCall()[0],
      0[charPtrCall()]
      );
}

void testArraySubscriptsWithAuto(int *p, int **pp) {
  int a[10];

  auto ap1 = a;

  foo(ap1[1]);  // expected-warning{{unchecked operation on raw buffer in expression}}

  auto ap2 = p;

  foo(ap2[1]);  // expected-warning{{unchecked operation on raw buffer in expression}}

  auto ap3 = pp;

  foo(ap3[1][1]); // expected-warning2{{unchecked operation on raw buffer in expression}}

  auto ap4 = *pp;

  foo(ap4[1]);  // expected-warning{{unchecked operation on raw buffer in expression}}
}

void testUnevaluatedContext(int * p) {
  //TODO: do not warn for unevaluated context
  foo(sizeof(p[1]),             // expected-warning{{unchecked operation on raw buffer in expression}}
      sizeof(decltype(p[1])));  // expected-warning{{unchecked operation on raw buffer in expression}}
}

void testQualifiedParameters(const int * p, const int * const q,
			     const int a[10], const int b[10][10],
			     int (&c)[10]) {
  foo(p[1], 1[p], p[-1],   // expected-warning3{{unchecked operation on raw buffer in expression}}
      q[1], 1[q], q[-1],   // expected-warning3{{unchecked operation on raw buffer in expression}}
      a[1],                // expected-warning{{unchecked operation on raw buffer in expression}}     `a` is of pointer type
      b[1][2]              // expected-warning2{{unchecked operation on raw buffer in expression}}     `b[1]` is of array type
      );
}

struct T {
  int a[10];
  int * b;
  struct {
    int a[10];
    int * b;
  } c;
};

typedef struct T T_t;

T_t funRetT();
T_t * funRetTStar();

void testStructMembers(struct T * sp, struct T s, T_t * sp2, T_t s2) {
  foo(sp->a[1],   // expected-warning{{unchecked operation on raw buffer in expression}}
     sp->b[1],     // expected-warning{{unchecked operation on raw buffer in expression}}
     sp->c.a[1],   // expected-warning{{unchecked operation on raw buffer in expression}}
     sp->c.b[1],   // expected-warning{{unchecked operation on raw buffer in expression}}
     s.a[1],       // expected-warning{{unchecked operation on raw buffer in expression}}
     s.b[1],       // expected-warning{{unchecked operation on raw buffer in expression}}
     s.c.a[1],     // expected-warning{{unchecked operation on raw buffer in expression}}
     s.c.b[1],     // expected-warning{{unchecked operation on raw buffer in expression}}
     sp2->a[1],    // expected-warning{{unchecked operation on raw buffer in expression}}
     sp2->b[1],    // expected-warning{{unchecked operation on raw buffer in expression}}
     sp2->c.a[1],  // expected-warning{{unchecked operation on raw buffer in expression}}
     sp2->c.b[1],  // expected-warning{{unchecked operation on raw buffer in expression}}
     s2.a[1],    // expected-warning{{unchecked operation on raw buffer in expression}}
     s2.b[1],      // expected-warning{{unchecked operation on raw buffer in expression}}
     s2.c.a[1],           // expected-warning{{unchecked operation on raw buffer in expression}}
     s2.c.b[1],           // expected-warning{{unchecked operation on raw buffer in expression}}
     funRetT().a[1],      // expected-warning{{unchecked operation on raw buffer in expression}}
     funRetT().b[1],      // expected-warning{{unchecked operation on raw buffer in expression}}
     funRetTStar()->a[1], // expected-warning{{unchecked operation on raw buffer in expression}}
     funRetTStar()->b[1]  // expected-warning{{unchecked operation on raw buffer in expression}}
  );
}

int garray[10];
int * gp = garray;
int gvar = gp[1];  // FIXME: file scope unsafe buffer access is not warned

void testLambdaCaptureAndGlobal(int * p) {
  int a[10];

  auto Lam = [p, a]() {
    return p[1] // expected-warning{{unchecked operation on raw buffer in expression}}
      + a[1] + garray[1] // expected-warning2{{unchecked operation on raw buffer in expression}}
      + gp[1];  // expected-warning{{unchecked operation on raw buffer in expression}}
  };
}

typedef T_t * T_ptr_t;

void testTypedefs(T_ptr_t p) {
  foo(p[1],      // expected-warning{{unchecked operation on raw buffer in expression}}
      p[1].a[1], // expected-warning2{{unchecked operation on raw buffer in expression}}
      p[1].b[1]  // expected-warning2{{unchecked operation on raw buffer in expression}}
      );
}

template<typename T, int N> T f(T t, T * pt, T a[N], T (&b)[N]) {
  foo(pt[1],    // expected-warning{{unchecked operation on raw buffer in expression}}
      a[1],     // expected-warning{{unchecked operation on raw buffer in expression}}
      b[1]);    // expected-warning{{unchecked operation on raw buffer in expression}}
  return &t[1]; // expected-warning{{unchecked operation on raw buffer in expression}}
}

// Testing pointer arithmetic for pointer-to-int, qualified multi-level
// pointer, pointer to a template type, and auto type
T_ptr_t getPtr();

template<typename T>
void testPointerArithmetic(int * p, const int **q, T * x) {
  int a[10];
  auto y = &a[0];

  foo(p + 1, 1 + p, p - 1,      // expected-warning3{{unchecked operation on raw buffer in expression}}
      *q + 1, 1 + *q, *q - 1,   // expected-warning3{{unchecked operation on raw buffer in expression}}
      x + 1, 1 + x, x - 1,      // expected-warning3{{unchecked operation on raw buffer in expression}}
      y + 1, 1 + y, y - 1,      // expected-warning3{{unchecked operation on raw buffer in expression}}
      getPtr() + 1, 1 + getPtr(), getPtr() - 1 // expected-warning3{{unchecked operation on raw buffer in expression}}
      );

  p += 1;  p -= 1;  // expected-warning2{{unchecked operation on raw buffer in expression}}
  *q += 1; *q -= 1; // expected-warning2{{unchecked operation on raw buffer in expression}}
  y += 1; y -= 1;   // expected-warning2{{unchecked operation on raw buffer in expression}}
  x += 1; x -= 1;   // expected-warning2{{unchecked operation on raw buffer in expression}}
}

void testTemplate(int * p) {
  int *a[10];
  foo(f(p, &p, a, a)[1]); // expected-warning{{unchecked operation on raw buffer in expression}}, \
                             expected-note{{in instantiation of function template specialization 'f<int *, 10>' requested here}}

  const int **q = const_cast<const int **>(&p);

  testPointerArithmetic(p, q, p); //expected-note{{in instantiation of function template specialization 'testPointerArithmetic<int>' requested here}}
}

void testPointerToMember() {
  struct S_t {
    int x;
    int * y;
  } S;

  int S_t::* p = &S_t::x;
  int * S_t::* q = &S_t::y;

  foo(S.*p,
      (S.*q)[1]);  // expected-warning{{unchecked operation on raw buffer in expression}}
}

// test that nested callable definitions are scanned only once
void testNestedCallableDefinition(int * p) {
  class A {
    void inner(int * p) {
      p++; // expected-warning{{unchecked operation on raw buffer in expression}}
    }

    static void innerStatic(int * p) {
      p++; // expected-warning{{unchecked operation on raw buffer in expression}}
    }

    void innerInner(int * p) {
      auto Lam = [p]() {
        int * q = p;
        q++;   // expected-warning{{unchecked operation on raw buffer in expression}}
        return *q;
      };
    }
  };

  auto Lam = [p]() {
    int * q = p;
    q++;  // expected-warning{{unchecked operation on raw buffer in expression}}
    return *q;
  };

  auto LamLam = [p]() {
    auto Lam = [p]() {
      int * q = p;
      q++;  // expected-warning{{unchecked operation on raw buffer in expression}}
      return *q;
    };
  };

  void (^Blk)(int*) = ^(int *p) {
    p++;   // expected-warning{{unchecked operation on raw buffer in expression}}
  };

  void (^BlkBlk)(int*) = ^(int *p) {
    void (^Blk)(int*) = ^(int *p) {
      p++;   // expected-warning{{unchecked operation on raw buffer in expression}}
    };
    Blk(p);
  };

  // lambda and block as call arguments...
  foo( [p]() { int * q = p;
              q++;  // expected-warning{{unchecked operation on raw buffer in expression}}
              return *q;
       },
       ^(int *p) { p++;   // expected-warning{{unchecked operation on raw buffer in expression}}
       }
     );
}

int testVariableDecls(int * p) {
  int * q = p++;      // expected-warning{{unchecked operation on raw buffer in expression}}
  int a[p[1]];        // expected-warning{{unchecked operation on raw buffer in expression}}
  int b = p[1];       // expected-warning{{unchecked operation on raw buffer in expression}}
  return p[1];        // expected-warning{{unchecked operation on raw buffer in expression}}
}
    
template<typename T> void fArr(T t[]) {
  foo(t[1]);   // expected-warning{{unchecked operation on raw buffer in expression}}
  T ar[8];
  foo(ar[5]);  // expected-warning{{unchecked operation on raw buffer in expression}}
}

template void fArr<int>(int t[]); // expected-note {{in instantiation of function template specialization 'fArr<int>' requested here}}

  int testReturn(int t[]) {
  return t[1]; // expected-warning{{unchecked operation on raw buffer in expression}}
 }
    
//FIXME: Array access warnings on 0-indices;ArraySubscriptGadget excludes 0 index for both raw pointers and arrays!
int testArrayAccesses(int n) {
          
  // auto deduced array type
  int cArr[2][3] = {{1, 2, 3}, {4, 5, 6}};
  int d = cArr[0][0];
  foo(cArr[0][0]);
  foo(cArr[1][2]);     // expected-warning2{{unchecked operation on raw buffer in expression}}
  auto cPtr = cArr[1][2];  // expected-warning2{{unchecked operation on raw buffer in expression}}
  foo(cPtr);
          
          // Typdefs
  typedef int A[3];
  const A tArr = {4, 5, 6};
  foo(tArr[0], tArr[1]);  // expected-warning{{unchecked operation on raw buffer in expression}}
  return cArr[0][1];  // expected-warning{{unchecked operation on raw buffer in expression}}
}
    
void testArrayPtrArithmetic(int x[]) {
  foo (x + 3); // expected-warning{{unchecked operation on raw buffer in expression}}
         
  int y[3] = {0, 1, 2};
  foo(y + 4); // expected-warning{{unchecked operation on raw buffer in expression}}
}

#endif
