// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -fsyntax-only -fstrict-flex-arrays=2 -verify=warn %s

@interface Flexible {
@public
  char flexible[];
}
@end

@interface Flexible0 {
@public
  char flexible[0];
}
@end

@interface Flexible1 {
@public
  char flexible[1];
}
@end

char readit(Flexible *p) { return p->flexible[2]; }
char readit0(Flexible0 *p) { return p->flexible[2]; }
char readit1(Flexible1 *p) { return p->flexible[2]; } // warn-warning {{array index 2 is past the end of the array (that has type 'char[1]')}}
