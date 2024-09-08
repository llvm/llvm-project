// RUN: %clang_analyze_cc1 -std=c23 -analyzer-checker=core,debug.ExprInspection -verify %s

// expected-no-diagnostics

int main() {
    const unsigned char SelfBytes[] = {
        #embed "embed.c"
    };
}
