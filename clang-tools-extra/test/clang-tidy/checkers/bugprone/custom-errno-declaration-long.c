// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

extern long errno; // should be ignored
