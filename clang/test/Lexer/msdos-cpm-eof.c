// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -Wmicrosoft %s

int x; 

// expected-warning@+1 {{treating Ctrl-Z as end-of-file is a Microsoft extension}}


I am random garbage after ^Z
