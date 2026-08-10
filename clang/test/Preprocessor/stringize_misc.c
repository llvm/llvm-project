#ifdef TEST1
// RUN: %clang_cc1 -E %s -DTEST1 | FileCheck -strict-whitespace %s

#define M(x, y) #x #y

M( f(1, 2), g((x=y++, y))) 
// CHECK: "f(1, 2)" "g((x=y++, y))"

M( {a=1 , b=2;} ) /* A semicolon is not a comma */ 
// CHECK: "{a=1" "b=2;}"

M( <, [ ) /* Passes the arguments < and [ */ 
// CHECK: "<" "["

M( (,), (...) ) /* Passes the arguments (,) and (...) */ 
// CHECK: "(,)" "(...)"

#define START_END(start, end) start c=3; end 

START_END( {a=1 , b=2;} ) /* braces are not parentheses */ 
// CHECK: {a=1 c=3; b=2;}

/* 
 * To pass a comma token as an argument it is 
 * necessary to write: 
 */ 
#define COMMA , 
 
M(a COMMA b, (a, b)) 
// CHECK: "a COMMA b" "(a, b)"

#endif

#ifdef TEST2
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST2

#define HASH #
#define INVALID() #
// expected-error@-1{{'#' is not followed by a macro parameter}}

#endif
