
// This pounds on macro expansion for performance reasons.  This is currently
// heavily constrained by darwin's malloc.

// Function-like macros.
#define A0(A, B) A B
#define A1(A, B) A0(A,B) A0(A,B) A0(A,B) A0(A,B) A0(A,B) A0(A,B)
#define A2(A, B) A1(A,B) A1(A,B) A1(A,B) A1(A,B) A1(A,B) A1(A,B)
#define A3(A, B) A2(A,B) A2(A,B) A2(A,B) A2(A,B) A2(A,B) A2(A,B)
#define A4(A, B) A3(A,B) A3(A,B) A3(A,B) A3(A,B) A3(A,B) A3(A,B)
#define A5(A, B) A4(A,B) A4(A,B) A4(A,B) A4(A,B) A4(A,B) A4(A,B)
#define A6(A, B) A5(A,B) A5(A,B) A5(A,B) A5(A,B) A5(A,B) A5(A,B)
#define A7(A, B) A6(A,B) A6(A,B) A6(A,B) A6(A,B) A6(A,B) A6(A,B)
#define A8(A, B) A7(A,B) A7(A,B) A7(A,B) A7(A,B) A7(A,B) A7(A,B)

A8(a, b)

