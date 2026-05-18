// RUN: %clang -fsyntax-only -std=c++20 -Xclang -verify %s

void Func(int x) {
    switch (x) {
        [[likely]] case 0:
        case 1: 
            int i = 3; // expected-note {{jump bypasses variable initialization}}
        case 2: // expected-error {{cannot jump from switch statement to this case label}}
            break;
    }
}
