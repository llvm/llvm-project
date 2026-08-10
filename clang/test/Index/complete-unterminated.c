typedef int Integer;

#if 0


#endif

/* blah */

void f0(const char*);
void f1(char);

const char *hello = "Hello, world";
const char a = 'a';

#define FOO(a, b) a b

FOO(int, x);

// RUN: c-index-test -code-completion-at=%s:5:1 -pedantic %s 2> %t.err | FileCheck %s
// RUN: not grep error %t.err
// CHECK: {TypedText Integer}
// RUN: c-index-test -code-completion-at=%s:8:6 -pedantic %s 2> %t.err
// RUN: not grep error %t.err
// RUN: c-index-test -code-completion-at=%s:10:28 -pedantic %s 2> %t.err
// RUN: not grep unterminated %t.err
// RUN: c-index-test -code-completion-at=%s:11:17 -pedantic %s 2> %t.err
// RUN: not grep unterminated %t.err
// RUN: c-index-test -code-completion-at=%s:18:10 -pedantic %s 2> %t.err
// RUN: not grep unterminated %t.err
