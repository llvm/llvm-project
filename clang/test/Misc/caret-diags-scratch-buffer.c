// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | not grep keyXXXX
// This should not show keyXXXX in the caret diag output.  This once
// happened because the two tokens ended up in the scratch buffer and
// the caret diag from the scratch buffer included the previous token.
#define M(name) \
    if (name ## XXXX != name ## _sb);

void foo() {
  int keyXXXX;
  M(key);
}

