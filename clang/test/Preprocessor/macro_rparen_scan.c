// RUN: %clang_cc1 -E %s | grep '^3 ;$'

/* Right paren scanning, hard case.  Should expand to 3. */
#define i(x) 3 
#define a i(yz 
#define b ) 
a b ) ; 

