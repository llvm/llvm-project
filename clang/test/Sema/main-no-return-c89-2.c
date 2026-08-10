/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -Wno-strict-prototypes -Wmain-return-type %s
 */

/* expected-no-diagnostics */

void exit(int);

int main() {
    if (1)
        exit(1);
}
