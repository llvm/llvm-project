// RUN: %clang_cc1 -fsyntax-only -Wmacro-mixed-operator -verify %s

// ------------------------------------------------------------
// Basic macro body precedence problem
// ------------------------------------------------------------
#define FOO 2+3
// 2+(3*4) == 14, not (2+3)*4 == 20
int n = FOO*4; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Macro argument precedence problem
// ------------------------------------------------------------
#define FOO_ARG(x) 2+x
int m = FOO_ARG(3*4); // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Operator entirely outside macros (should NOT warn)
// ------------------------------------------------------------
int k = 2 + 3 * 4; // no-warning

// ------------------------------------------------------------
// Operator comes from macro argument only (should NOT warn)
// ------------------------------------------------------------
#define ID(x) x
int p = ID(2+3)*4; // no-warning

// ------------------------------------------------------------
// Macro with proper parentheses (should NOT warn)
// ------------------------------------------------------------
#define SAFE_ADD (2+3)
int q = SAFE_ADD*4; // no-warning

#define SAFE_ARG(x) (2+(x))
int r = SAFE_ARG(3*4); // no-warning

// ------------------------------------------------------------
// Nested macro expansion
// ------------------------------------------------------------
#define INNER 2+3
#define OUTER INNER
int s = OUTER*4; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Macro producing multiplication interacting with external +
// ------------------------------------------------------------
#define MUL 2*3
int t = MUL+4; // no-warning

// ------------------------------------------------------------
// Macro with multiple operators
// ------------------------------------------------------------
#define MIXED 1+2*3
int u = MIXED+4; // no-warning

// ------------------------------------------------------------
// Macro argument containing another macro
// ------------------------------------------------------------
#define ADD(x) 2+x
#define VALUE 3*4
int v = ADD(VALUE); // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Macro where entire expression stays inside expansion
// ------------------------------------------------------------
#define FULL (2+3)
int w = FULL; // no-warning

// ------------------------------------------------------------
// Operator both operands inside macro body
// ------------------------------------------------------------
#define ADD_TWO (2+3)
int x = ADD_TWO; // no-warning

// ------------------------------------------------------------
// Chained macro expansions
// ------------------------------------------------------------
#define A 2+3
#define B A
#define C B
int y = C*4; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Unary operator on macro (should warn)
// ------------------------------------------------------------
#define VAL 2+3
int un1 = -VAL; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Macro used as function argument (should NOT warn)
// ------------------------------------------------------------
extern int func(int, int);
#define EXPR 2+3
void test_func_args(void) {
int fa1 = func(EXPR, 1); // no-warning
int fa2 = func(1, EXPR); // no-warning
}

// ------------------------------------------------------------
// Macro on both sides of operator (should NOT warn)
// ------------------------------------------------------------
#define TWO 2
#define THREE 3
int bb1 = TWO + THREE; // no-warning

// ------------------------------------------------------------
// Macro result used in comparison (lower precedence, should NOT warn)
// ------------------------------------------------------------
#define SUM 2+3
int cmp1 = SUM == 5; // no-warning

// ------------------------------------------------------------
// Macro result used in logical op (lower precedence, should NOT warn)
// ------------------------------------------------------------
int cmp2 = SUM && 1; // no-warning
int cmp3 = SUM || 0; // no-warning

// ------------------------------------------------------------
// Higher-precedence op outside, macro has lower-precedence op inside
// ------------------------------------------------------------
#define ADDM 1+2
int ho1 = ADDM * 3; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Ternary condition uses macro (should NOT warn)
// ------------------------------------------------------------
#define COND 1+1
int tern1 = COND ? 1 : 0; // no-warning

// ------------------------------------------------------------
// Macro in assignment RHS (should NOT warn)
// ------------------------------------------------------------
#define RVAL 3+4
int asgn2 = RVAL; // no-warning

// ------------------------------------------------------------
// Macro argument with same-precedence op (should warn)
// ------------------------------------------------------------
#define WRAP(x) 1+x
int sa1 = WRAP(2+3); // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Double-nested argument macro
// ------------------------------------------------------------
#define OUTER2(x) 10+x
#define INNER2(x) x*2
int dn1 = OUTER2(INNER2(3)); // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Macro used twice in same expression
// ------------------------------------------------------------
#define BASE 1+2
int tw1 = BASE * BASE; // expected-warning 2 {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Parenthesized macro argument (should NOT warn)
// ------------------------------------------------------------
#define ADDP(x) 5+(x)
int pa1 = ADDP(3*2); // no-warning

// ------------------------------------------------------------
// Shift operators interacting with macro addition: existing precedence warning
// ------------------------------------------------------------
#define SHIFT_BASE 1+2
int sh1 = SHIFT_BASE << 1; // expected-warning {{operator '<<' has lower precedence than '+'; '+' will be evaluated first}} expected-note {{place parentheses around the '+' expression to silence this warning}}
int sh2 = SHIFT_BASE >> 1; // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} expected-note {{place parentheses around the '+' expression to silence this warning}}

// ------------------------------------------------------------
// Bitwise AND with macro addition (should NOT warn)
// ------------------------------------------------------------
#define BVAL 2+3
int bw1 = BVAL & 0xFF; // no-warning

// ------------------------------------------------------------
// Subtraction in macro body with multiplication outside (should warn)
// ------------------------------------------------------------
#define SUBM 5-2
int sub1 = SUBM * 3; // expected-warning {{operator '-' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Division in macro body with addition outside (should NOT warn)
// ------------------------------------------------------------
#define DIVM 6/2
int div1 = DIVM + 1; // no-warning

// ------------------------------------------------------------
// Deeply chained: DD->DC->DB->DA, multiply outside (should warn)
// ------------------------------------------------------------
#define DA 1+2
#define DB DA
#define DC DB
#define DD DC
int dc1 = DD * 5; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}

// ------------------------------------------------------------
// Nested macro expansion: FileID approach would incorrectly think
// the operand escapes, getImmediateMacroCallerLoc handles it correctly
// ------------------------------------------------------------
#define INNER_MACRO 2+3
#define OUTER_MACRO INNER_MACRO
int nested_test = OUTER_MACRO*4; // expected-warning {{operator '+' in macro expansion has operand outside the macro; operator precedence may be different than expected}}