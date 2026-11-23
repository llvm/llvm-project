// RUN: %clang_cc1 -E -verify %s

// C++ [lex.icon]p4 and C 6.4.4.1p2 + 6.4.4.2p7 both require C and C++ to
// validate the integer constant value when converting a preprocessing token
// into a token for semantic analysis, even within the preprocessor itself.

// Plain integer constant.
#if 999999999999999999999 // expected-error {{integer literal is too large to be represented in any integer type}}
#endif

// These cases were previously incorrectly accepted. See GH134658.

// Integer constant in an unevaluated branch of a conditional.
#if 1 ? 1 : 999999999999999999999 // expected-error {{integer literal is too large to be represented in any integer type}}
#endif

// Integer constant in an unevaluated operand of a logical operator.
#if 0 && 999999999999999999999 // expected-error {{integer literal is too large to be represented in any integer type}}
#endif

#if 1 || 999999999999999999999 // expected-error {{integer literal is too large to be represented in any integer type}}
#endif

// Make sure we also catch it in an elif condition.
#if 0
#elif 1 || 999999999999999999999 // expected-error {{integer literal is too large to be represented in any integer type}}
#endif

// However, if the block is skipped entirely, then it doesn't matter how
// invalid the constant value is.
#if 0
int x = 999999999999999999999;

#if 999999999999999999999
#endif

#if 0 && 999999999999999999999
#endif

#endif
