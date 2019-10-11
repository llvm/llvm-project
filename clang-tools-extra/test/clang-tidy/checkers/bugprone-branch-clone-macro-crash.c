// RUN: %check_clang_tidy %s bugprone-branch-clone %t
int x = 0;
int y = 1;
#define a(b, c) \
  typeof(b) d;  \
  if (b)        \
    d = b;      \
  else if (c)   \
    d = b;

f() {
  // CHECK-MESSAGES: warning: repeated branch in conditional chain [bugprone-branch-clone]
  a(x, y)
}
