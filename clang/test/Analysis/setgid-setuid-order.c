// RUN: %clang_analyze_cc1 -analyzer-checker=core,security.SetgidSetuidOrder -verify %s

typedef int uid_t;
typedef int gid_t;

int setuid(uid_t);
int setgid(gid_t);
int seteuid(uid_t);
int setegid(gid_t);
int setreuid(uid_t, uid_t);
int setregid(gid_t, gid_t);
int setresuid(uid_t, uid_t, uid_t);
int setresgid(gid_t, gid_t, gid_t);

uid_t getuid();
gid_t getgid();



void correct_order() {
  // A correct revocation sequence starts here.
  if (setgid(getgid()) == -1)
    return;
  if (setuid(getuid()) == -1)
    return;
  // No warning for the following setgid statement.
  // The previous setgid and setuid calls are a correct privilege revocation
  // sequence. The checker does not care about the following statements (except
  // if a wrong setuid-setgid sequence follows again).
  if (setgid(getgid()) == -1)
    return;
}

void incorrect_after_correct() {
  if (setgid(getgid()) == -1)
    return;
  if (setuid(getuid()) == -1)
    return;
  // Incorrect sequence starts here.
  if (setuid(getuid()) == -1)
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

void incorrect_order() {
  if (setuid(getuid()) == -1)
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
  if (setgid(getgid()) == -1)
    return;
}

void warn_at_second_time() {
  if (setuid(getuid()) == -1)
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
  if (setuid(getuid()) == -1)
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

uid_t f_uid();
gid_t f_gid();

void setuid_other() {
  if (setuid(f_uid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setgid_other() {
  if (setuid(getuid()) == -1)
    return;
  if (setgid(f_gid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setuid_other_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setuid(f_uid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setgid_with_getuid() {
  if (setuid(getuid()) == -1)
    return;
  // add a clang-tidy check for this case?
  if (setgid(getuid()) == -1)
    return;
}

void setuid_with_getgid() {
  // add a clang-tidy check for this case?
  if (setuid(getgid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

int f_setuid() {
  return setuid(getuid());
}

int f_setgid() {
  return setgid(getgid()); // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
}

void function_calls() {
  if (f_setuid() == -1)
    return;
  if (f_setgid() == -1)
    return;
}

void seteuid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (seteuid(getuid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setegid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setegid(getgid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setreuid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setreuid(getuid(), getuid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setregid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setregid(getgid(), getgid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setresuid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setresuid(getuid(), getuid(), getuid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void setresgid_between() {
  if (setuid(getuid()) == -1)
    return;
  if (setresgid(getgid(), getgid(), getgid()) == -1)
    return;
  if (setgid(getgid()) == -1)
    return;
}

void getgid_getuid_between() {
  if (setuid(getuid()) == -1)
    return;
  (void)getgid();
  (void)getuid();
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

void stored_getgid_getuid() {
  // possible future improvement: detect this case
  uid_t u = getuid();
  gid_t g = getgid();
  if (setuid(u) == -1)
    return;
  if (setgid(g) == -1) // no warning
    return;
}

void f_extern();

void other_unknown_function_between() {
  if (setuid(getuid()) == -1)
    return;
  f_extern();
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

void setuid_error_case() {
  if (setuid(getuid()) == -1) {
    // No warning if we know that the first setuid call has failed.
    (void)setgid(getgid());
    return;
  }
  (void)setgid(getgid()); // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
}

void setuid_success_case() {
  if (setuid(getuid()) == 0) {
    if (setgid(getgid()) == 0) { // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    }
  }
}

void incorrect_order_compare_zero() {
  if (setuid(getuid()) != 0)
    return;
  (void)setgid(getgid()); // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
}

void setuid_error_case_compare_zero() {
  if (setuid(getuid()) != 0) {
    // No warning if we know that the first setuid call has failed.
    (void)setgid(getgid());
    return;
  }
}

void incorrect_order_compare_other() {
  if (setuid(getuid()) == -2) {
    // This is a case for improvement:
    // The checker does not recognize that this is an invalid error check,
    // but this is really another type of bug not related to this checker.
    (void)setgid(getgid()); // warning should appear here
    return;
  }
  if (setgid(getgid()) == -2) { // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
  }
}

const int FAIL = -1;

void incorrect_order_compare_var() {
  if (setuid(getuid()) == FAIL)
    return;
  (void)setgid(getgid()); // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
}
