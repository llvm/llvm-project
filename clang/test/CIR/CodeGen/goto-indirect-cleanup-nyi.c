// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s

// A `goto *p` that leaves a scope needing cleanup (here a VLA stack restore)
// must run that cleanup on the branch.  That is not implemented yet, so it is
// reported rather than lowered to a branch that skips the cleanup.
int vla(int n) {
  int a[n];
  void *p = &&done;
  // expected-error@+1 {{indirect goto across a cleanup scope}}
  goto *p;
done:
  return a[0];
}
