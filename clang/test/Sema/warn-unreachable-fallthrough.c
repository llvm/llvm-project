// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x -Wunreachable-code-fallthrough %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x -Wunreachable-code %s
// RUN: %clang_cc1 -fsyntax-only -verify=code -std=c2x -Wunreachable-code -Wno-unreachable-code-fallthrough %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x -Wno-unreachable-code -Wunreachable-code-fallthrough %s

int n;
void f(void){
     switch (n){
         [[fallthrough]]; // expected-warning{{fallthrough annotation in unreachable code}}
                          // code-warning@-1{{never be executed}}
         case 1:;
     }
}
