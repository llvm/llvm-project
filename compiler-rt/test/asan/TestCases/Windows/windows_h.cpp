// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t

// Just make sure we can parse <windows.h>
#include <windows.h>

int main() {}
