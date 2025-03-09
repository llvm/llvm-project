// RUN: %clang_cc1 -std=c89 -verify=c89 %s

// From: https://github.com/llvm/llvm-project/issues/15522#issue-1071059939
int logf = 5; // this is fine

// redefinition because log as a symbol exists in C89
int log = 6; // #1
int main() {
return 0;
}

// c89-error@#1 {{redefinition of 'log' as different kind of symbol}}
// c89-note@#1 {{unguarded header; consider using #ifdef guards or #pragma once}}
// c89-note@#1 {{previous definition}}
