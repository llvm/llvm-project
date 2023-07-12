/*
  RUN: %clang_cc1 -E -ftrigraphs %s | grep bar
  RUN: %clang_cc1 -E -ftrigraphs %s | grep foo
  RUN: %clang_cc1 -E -ftrigraphs %s | not grep qux
  RUN: %clang_cc1 -E -ftrigraphs %s | not grep xyz
  RUN: %clang_cc1 -fsyntax-only -ftrigraphs -verify %s  
*/

// This is a simple comment, /*/ does not end a comment, the trailing */ does.
int i = /*/ */ 1;

/* qux

next comment ends with normal escaped newline:
*/

/* expected-warning {{escaped newline}} expected-warning {{backslash and newline}}  *\  
/

int bar /* expected-error {{expected ';' after top level declarator}} */

/* xyz

next comment ends with a trigraph escaped newline: */

/* expected-warning {{escaped newline between}}   expected-warning {{backslash and newline separated by space}}    expected-warning {{trigraph ends block comment}}   *??/    
/

foo


// We should not get warnings about trigraphs in comments:
// '????'
/* ???? */

// PR50456: multiple escaped newlines in one */.
/*
 *\
??/
??/  
\  
/
// expected-warning@-5 {{escaped newline}}
// expected-warning@-4 {{separated by space}}
// expected-warning@-6 {{trigraph ends block comment}}
