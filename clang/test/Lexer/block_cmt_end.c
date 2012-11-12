/*
  RUN: %clang_cc1 -E -trigraphs %s | grep bar
  RUN: %clang_cc1 -E -trigraphs %s | grep foo
  RUN: %clang_cc1 -E -trigraphs %s | not grep qux
  RUN: %clang_cc1 -E -trigraphs %s | not grep xyz
  RUN: %clang_cc1 -fsyntax-only -trigraphs -verify %s  
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


// rdar://6060752 - We should not get warnings about trigraphs in comments:
// '????'
/* ???? */
