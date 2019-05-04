// RUN: %clang_cc1 -E %s -pedantic -std=c++2a | FileCheck -strict-whitespace %s

#define LPAREN ( 
#define RPAREN ) 

#define A0 expandedA0
#define A1  expandedA1 A0
#define A2  expandedA2 A1
#define A3  expandedA3 A2

#define A() B LPAREN )
#define B() C LPAREN )
#define C() D LPAREN )


#define F(x, y) x + y 
#define ELLIP_FUNC(...) __VA_OPT__(__VA_ARGS__)

1: ELLIP_FUNC(F, LPAREN, 'a', 'b', RPAREN); 
2: ELLIP_FUNC(F LPAREN 'a', 'b' RPAREN); 
#undef F
#undef ELLIP_FUNC

// CHECK: 1: F, (, 'a', 'b', );
// CHECK: 2: 'a' + 'b';

#define F(...) f(0 __VA_OPT__(,) __VA_ARGS__)
3: F(a, b, c) // replaced by f(0, a, b, c) 
4: F() // replaced by f(0)

// CHECK: 3: f(0 , a, b, c) 
// CHECK: 4: f(0 )
#undef F

#define G(X, ...) f(0, X __VA_OPT__(,) __VA_ARGS__)

5: G(a, b, c) // replaced by f(0, a , b, c) 
6: G(a) // replaced by f(0, a) 
7: G(a,) // replaced by f(0, a) 
7.1: G(a,,)


// CHECK: 5: f(0, a , b, c) 
// CHECK: 6: f(0, a ) 
// CHECK: 7: f(0, a ) 
// CHECK: 7.1: f(0, a , ,)
#undef G 

#define HT_B() TONG

#define F(x, ...) HT_ ## __VA_OPT__(x x A()  #x)

8: F(1)
9: F(A(),1)

// CHECK: 8: HT_
// CHECK: 9: TONG C ( ) B ( ) "A()"
#undef HT_B
#undef F

#define F(a,...) #__VA_OPT__(A1 a)

10: F(A())
11: F(A1 A(), 1)
// CHECK: 10: ""
// CHECK: 11: "A1 expandedA1 expandedA0 B ( )"
#undef F


#define F(a,...) a ## __VA_OPT__(A1 a) ## __VA_ARGS__ ## a
12.0: F()
12: F(,)
13: F(B,)
// CHECK: 12.0: 
// CHECK: 12: 
// CHECK: 13: BB 
#undef F

#define F(...) #__VA_OPT__()  X ## __VA_OPT__()  #__VA_OPT__(        )

14: F()
15: F(1)

// CHECK: 14: "" X ""
// CHECK: 15: "" X ""

#undef F

#define SDEF(sname, ...) S sname __VA_OPT__(= { __VA_ARGS__ })

16: SDEF(foo); // replaced by S foo; 
17: SDEF(bar, 1, 2); // replaced by S bar = { 1, 2 }; 

// CHECK: 16: S foo ;
// CHECK: 17: S bar = { 1, 2 }; 
#undef SDEF

#define F(a,...) A() #__VA_OPT__(A3 __VA_ARGS__ a ## __VA_ARGS__ ## a ## C A3) A()

18: F()
19: F(,)
20: F(,A3)
21: F(A3, A(),A0)


// CHECK: 18: B ( ) "" B ( ) 
// CHECK: 19: B ( ) "" B ( ) 
// CHECK: 20: B ( ) "A3 expandedA3 expandedA2 expandedA1 expandedA0 A3C A3" B ( )
// CHECK: 21: B ( ) "A3 B ( ),expandedA0 A3A(),A0A3C A3" B ( )

#undef F

#define F(a,...) A() #__VA_OPT__(A3 __VA_ARGS__ a ## __VA_ARGS__ ## a ## C A3) a __VA_OPT__(A0 __VA_ARGS__ a ## __VA_ARGS__ ## a ## C A0) A()

22: F()
23: F(,)
24: F(,A0)
25: F(A0, A(),A0)


// CHECK: 22: B ( ) "" B ( ) 
// CHECK: 23: B ( ) "" B ( ) 
// CHECK: 24: B ( ) "A3 expandedA0 A0C A3" expandedA0 expandedA0 A0C expandedA0 B ( )
// CHECK: 25: B ( ) "A3 B ( ),expandedA0 A0A(),A0A0C A3" expandedA0 expandedA0 C ( ),expandedA0 A0A(),A0A0C expandedA0 B ( )

#undef F

#define F(a,...)  __VA_OPT__(B a ## a) ## 1
#define G(a,...)  __VA_OPT__(B a) ## 1
26: F(,1)
26_1: G(,1)
// CHECK: 26: B 1
// CHECK: 26_1: B 1
#undef F
#undef G

#define F(a,...)  B ## __VA_OPT__(a 1) ## 1
#define G(a,...)  B ## __VA_OPT__(a ## a 1) ## 1

27: F(,1)
27_1: F(A0,1)
28: G(,1)
// CHECK: 27: B 11
// CHECK: 27_1: BexpandedA0 11
// CHECK: 28: B 11

#undef F
#undef G
