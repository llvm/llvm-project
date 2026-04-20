// RUN: %clang_cc1 -verify -std=c89 %s

// See https://github.com/llvm/llvm-project/issues/15522

// logf doesn't exist in C89
int logf = 5;

// But log does
int log = 6; // expected-error {{redefinition of 'log' as different kind of symbol}} \
		expected-note {{unguarded header; consider using #ifdef guards or #pragma once}} \
		expected-note {{previous definition is here}}

int main() {
    return logf;
}
