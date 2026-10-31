// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

typedef int myint;
myint errno; // should be ignored
