// RUN: %clang_cc1 -verify -fsyntax-only %s

int __attribute__((not_tail_called)) foo1(int a) {// expected-note {{'not_tail_called' attribute prevents being called as a tail call}}
    return a + 1;  
}


int foo2(int a) {
    [[clang::musttail]] 
    return foo1(a);  // expected-error {{cannot perform a tail call to function 'foo1' because its signature is incompatible with the calling function}} 
}

int main() {
    int result = foo2(10);      
    return 0;
}

