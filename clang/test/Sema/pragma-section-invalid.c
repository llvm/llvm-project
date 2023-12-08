// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-apple-darwin

// expected-error@+1 {{argument to 'section' attribute is not valid for this target: mach-o section specifier requires a segment and section separated by a comma}}
#pragma data_seg(".my_const")
int a = 1;
#pragma data_seg("__THINGY,thingy")
int b = 1;
