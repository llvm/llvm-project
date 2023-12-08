// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s

#define LPAREN ( 
#define RPAREN ) 
#define F(x, y) x + y 
#define ELLIP_FUNC(...) __VA_ARGS__ 

1: ELLIP_FUNC(F, LPAREN, 'a', 'b', RPAREN); /* 1st invocation */ 
2: ELLIP_FUNC(F LPAREN 'a', 'b' RPAREN); /* 2nd invocation */ 

// CHECK: 1: F, (, 'a', 'b', );
// CHECK: 2: 'a' + 'b';

