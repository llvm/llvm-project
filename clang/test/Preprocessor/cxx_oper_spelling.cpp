// RUN: %clang_cc1 -E %s | grep 'a: "and"'

#define X(A) #A

// C++'03 2.5p2: "In all respects of the language, each alternative 
// token behaves the same, respectively, as its primary token, 
// except for its spelling"
//
// This should be spelled as 'and', not '&&'
a: X(and)

