#if GC
# define GCONST const
#else
# define GCONST
#endif

// gconst-note@8 {{variable 'glb' declared const here}}
GCONST int glb = 5;


// Check various correct prefix spellings and combinations.
//
// RUN: %clang_cc1             -DGC           -verify=gconst                 %s
// RUN: %clang_cc1 -Wcast-qual      -DLC      -verify=lconst                 %s
// RUN: %clang_cc1                       -DSC -verify=expected               %s
// RUN: %clang_cc1                       -DSC -verify                        %s
// RUN: %clang_cc1                       -DSC -verify -verify                %s
// RUN: %clang_cc1                            -verify=nconst                 %s
// RUN: %clang_cc1                            -verify=n-const                %s
// RUN: %clang_cc1                            -verify=n_const                %s
// RUN: %clang_cc1                            -verify=NConst                 %s
// RUN: %clang_cc1                            -verify=NConst2                %s
// RUN: %clang_cc1 -Wcast-qual -DGC -DLC      -verify=gconst,lconst          %s
// RUN: %clang_cc1 -Wcast-qual -DGC -DLC -DSC -verify=gconst,lconst,expected %s
// RUN: %clang_cc1 -Wcast-qual -DGC -DLC      -verify=gconst -verify=lconst  %s
// RUN: %clang_cc1 -Wcast-qual -DGC -DLC -DSC -verify=gconst,lconst -verify  %s
// RUN: %clang_cc1             -DGC      -DSC -verify -verify=gconst -verify %s
//
// Duplicate prefixes.
// RUN: %clang_cc1 -Wcast-qual -DGC -DLC      -verify=gconst,lconst,gconst         %s
// RUN: %clang_cc1             -DGC           -verify=gconst -verify=gconst,gconst %s
// RUN: %clang_cc1                       -DSC -verify=expected -verify=expected    %s
// RUN: %clang_cc1                       -DSC -verify -verify=expected             %s
//
// Various tortured cases: multiple directives with different prefixes per
// line, prefixes used as comments, prefixes prefixing prefixes, and prefixes
// with special suffixes.
// RUN: %clang_cc1 -Wcast-qual      -DLC      -verify=foo                    %s
// RUN: %clang_cc1                       -DSC -verify=bar                    %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo,bar                %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=bar,foo                %s
// RUN: %clang_cc1                       -DSC -verify=foo-bar                %s
// RUN: %clang_cc1 -Wcast-qual      -DLC      -verify=bar-foo                %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo,foo-bar            %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo-bar,foo            %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=bar,bar-foo            %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=bar-foo,bar            %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo-bar,bar-foo        %s
// RUN: %clang_cc1                       -DSC -verify=foo-warning            %s
// RUN: %clang_cc1 -Wcast-qual      -DLC      -verify=bar-warning-re         %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo,foo-warning        %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=foo-warning,foo        %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=bar,bar-warning-re     %s
// RUN: %clang_cc1 -Wcast-qual      -DLC -DSC -verify=bar-warning-re,bar     %s


// Check invalid prefixes.  Check that there's no additional output, which
// might indicate that diagnostic verification became enabled even though it
// was requested incorrectly.  Check that prefixes are reported in command-line
// order.
//
// RUN: not %clang_cc1 -verify=5abc,-xy,foo,_k -verify='#a,b$' %s 2> %t
// RUN: FileCheck --check-prefixes=ERR %s < %t
//
// ERR-NOT:  {{.}}
// ERR:      error: invalid value '5abc' in '-verify='
// ERR-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// ERR-NEXT: error: invalid value '-xy' in '-verify='
// ERR-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// ERR-NEXT: error: invalid value '_k' in '-verify='
// ERR-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// ERR-NEXT: error: invalid value '#a' in '-verify='
// ERR-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// ERR-NEXT: error: invalid value 'b$' in '-verify='
// ERR-NEXT: note: -verify prefixes must start with a letter and contain only alphanumeric characters, hyphens, and underscores
// ERR-NOT:  {{.}}


// Check that our test code actually has expected diagnostics when there's no
// -verify.
//
// RUN: not %clang_cc1 -Wcast-qual -DGC -DLC -DSC %s 2> %t
// RUN: FileCheck --check-prefix=ALL %s < %t
//
// ALL: cannot assign to variable 'glb' with const-qualified type 'const int'
// ALL: variable 'glb' declared const here
// ALL: cast from 'const int *' to 'int *' drops const qualifier
// ALL: initializing 'int *' with an expression of type 'const int *' discards qualifiers


#if LC
# define LCONST const
#else
# define LCONST
#endif

#if SC
# define SCONST const
#else
# define SCONST
#endif

void foo(void) {
  LCONST int loc = 5;
  SCONST static int sta = 5;
  // We don't actually expect 1-2 occurrences of this error.  We're just
  // checking the parsing.
  glb = 6; // gconst-error1-2 {{cannot assign to variable 'glb' with const-qualified type 'const int'}}
  *(int*)(&loc) = 6; // lconst-warning {{cast from 'const int *' to 'int *' drops const qualifier}}
  ; // Code, comments, and many directives with different prefixes per line, including cases where some prefixes (foo and bar) prefix others (such as foo-bar and bar-foo), such that some prefixes appear as normal comments and some have special suffixes (-warning and -re): foo-warning@-1 {{cast from 'const int *' to 'int *' drops const qualifier}} foo-bar-warning@+1 {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}} foo-warning-warning@+1 {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}} bar-warning-re-warning@-1 {{cast from 'const int *' to 'int *' drops const qualifier}} bar-foo-warning@-1 {{cast from 'const int *' to 'int *' drops const qualifier}} bar-warning@+1 {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}}
  int *p = &sta; // expected-warning {{initializing 'int *' with an expression of type 'const int *' discards qualifiers}}
}

// nconst-no-diagnostics
// n-const-no-diagnostics
// n_const-no-diagnostics
// NConst-no-diagnostics
// NConst2-no-diagnostics
