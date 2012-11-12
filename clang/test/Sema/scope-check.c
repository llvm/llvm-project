// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=gnu99 %s -Wno-unreachable-code

int test1(int x) {
  goto L;    // expected-error{{goto into protected scope}}
  int a[x];  // expected-note {{jump bypasses initialization of variable length array}}
  int b[x];  // expected-note {{jump bypasses initialization of variable length array}}
  L:
  return sizeof a;
}

int test2(int x) {
  goto L;            // expected-error{{goto into protected scope}}
  typedef int a[x];  // expected-note {{jump bypasses initialization of VLA typedef}}
  L:
  return sizeof(a);
}

void test3clean(int*);

int test3() {
  goto L;            // expected-error{{goto into protected scope}}
int a __attribute((cleanup(test3clean))); // expected-note {{jump bypasses initialization of variable with __attribute__((cleanup))}}
L:
  return a;
}

int test4(int x) {
  goto L;       // expected-error{{goto into protected scope}}
int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  test4(x);
L:
  return sizeof a;
}

int test5(int x) {
  int a[x];
  test5(x);
  goto L;  // Ok.
L:
  goto L;  // Ok.
  return sizeof a;
}

int test6() { 
  // just plain invalid.
  goto x;  // expected-error {{use of undeclared label 'x'}}
}

void test7(int x) {
  switch (x) {
  case 1: ;
    int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  case 2:           // expected-error {{switch case is in protected scope}}
    a[1] = 2;
    break;
  }
}

int test8(int x) {
  // For statement.
  goto L2;     // expected-error {{goto into protected scope}}
  for (int arr[x];   // expected-note {{jump bypasses initialization of variable length array}}  
       ; ++x)
    L2:;

  // Statement expressions.
  goto L3;   // expected-error {{goto into protected scope}}
  int Y = ({  int a[x];   // expected-note {{jump bypasses initialization of variable length array}}  
           L3: 4; });
  
  goto L4; // expected-error {{goto into protected scope}}
  {
    int A[x],  // expected-note {{jump bypasses initialization of variable length array}}
        B[x];  // expected-note {{jump bypasses initialization of variable length array}}
  L4: ;
  }
  
  {
  L5: ;// ok
    int A[x], B = ({ if (x)
                       goto L5;
                     else 
                       goto L6;
                   4; }); 
  L6:; // ok.
    if (x) goto L6; // ok
  }
  
  {
  L7: ;// ok
    int A[x], B = ({ if (x)
                       goto L7;
                     else 
                       goto L8;  // expected-error {{goto into protected scope}}
                     4; }),
        C[x];   // expected-note {{jump bypasses initialization of variable length array}}
  L8:; // bad
  }
 
  {
  L9: ;// ok
    int A[({ if (x)
               goto L9;
             else
               // FIXME:
               goto L10;  // fixme-error {{goto into protected scope}}
           4; })];
  L10:; // bad
  }
  
  {
    // FIXME: Crashes goto checker.
    //goto L11;// ok
    //int A[({   L11: 4; })];
  }
  
  {
    goto L12;
    
    int y = 4;   // fixme-warn: skips initializer.
  L12:
    ;
  }
  
  // Statement expressions 2.
  goto L1;     // expected-error {{goto into protected scope}}
  return x == ({
                 int a[x];   // expected-note {{jump bypasses initialization of variable length array}}  
               L1:
                 42; });
}

void test9(int n, void *P) {
  int Y;
  int Z = 4;
  goto *P;  // expected-error {{indirect goto might cross protected scopes}}

L2: ;
  int a[n]; // expected-note {{jump bypasses initialization of variable length array}}

L3:         // expected-note {{possible target of indirect goto}}
L4:  
  goto *P;
  goto L3;  // ok
  goto L4;  // ok
  
  void *Ptrs[] = {
    &&L2,
    &&L3
  };
}

void test10(int n, void *P) {
  goto L0;     // expected-error {{goto into protected scope}}
  typedef int A[n];  // expected-note {{jump bypasses initialization of VLA typedef}}
L0:
  
  goto L1;      // expected-error {{goto into protected scope}}
  A b, c[10];        // expected-note 2 {{jump bypasses initialization of variable length array}}
L1:
  goto L2;     // expected-error {{goto into protected scope}}
  A d[n];      // expected-note {{jump bypasses initialization of variable length array}}
L2:
  return;
}

void test11(int n) {
  void *P = ^{
    switch (n) {
    case 1:;
    case 2: 
    case 3:;
      int Arr[n]; // expected-note {{jump bypasses initialization of variable length array}}
    case 4:       // expected-error {{switch case is in protected scope}}
      return;
    }
  };
}


// TODO: When and if gotos are allowed in blocks, this should work.
void test12(int n) {
  void *P = ^{
    goto L1;
  L1:
    goto L2;
  L2:
    goto L3;    // expected-error {{goto into protected scope}}
    int Arr[n]; // expected-note {{jump bypasses initialization of variable length array}}
  L3:
    goto L4;
  L4: return;
  };
}

void test13(int n, void *p) {
  int vla[n];
  goto *p;
 a0: ;
  static void *ps[] = { &&a0 };
}

int test14(int n) {
  static void *ps[] = { &&a0, &&a1 };
  if (n < 0)
    goto *&&a0;

  if (n > 0) {
    int vla[n];
   a1:
    vla[n-1] = 0;
  }
 a0:
  return 0;
}


// PR8473: IR gen can't deal with indirect gotos past VLA
// initialization, so that really needs to be a hard error.
void test15(int n, void *pc) {
  static const void *addrs[] = { &&L1, &&L2 };

  goto *pc; // expected-error {{indirect goto might cross protected scope}}

 L1:
  {
    char vla[n]; // expected-note {{jump bypasses initialization}}
   L2: // expected-note {{possible target}}
    vla[0] = 'a';
  }
}

// rdar://9024687
int test16(int [sizeof &&z]); // expected-error {{use of address-of-label extension outside of a function body}}
