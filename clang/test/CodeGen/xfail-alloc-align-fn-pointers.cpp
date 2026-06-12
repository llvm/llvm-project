
// RUN: %clang_cc1 %s

// FIXME: These should not crash!
// XFAIL: *

void aa_fn_ptr(char* (*member)(char*)  __attribute__((alloc_align(1))));

struct Test;
void aa_member_fn_ptr(char* (Test::*member)(char*)  __attribute__((alloc_align(1))));
