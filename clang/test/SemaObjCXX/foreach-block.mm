// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s
// rdar://8295106

int main() {
id array;

    for (int (^b)(void) in array) {
        if (b() == 10000) {
            return 1;
        }
    }

    int (^b)(void) in array; // expected-error {{expected ';' at end of declaration}}
}
