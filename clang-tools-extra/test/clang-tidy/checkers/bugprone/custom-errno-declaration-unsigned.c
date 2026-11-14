// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

extern unsigned errno; // should be ignored
