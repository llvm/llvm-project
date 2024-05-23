// RUN: %clang_analyze_cc1 -analyzer-checker=core,security.SetgidSetuidOrder -analyzer-output=text -verify %s

typedef int uid_t;
typedef int gid_t;

int setuid(uid_t);
int setgid(gid_t);

uid_t getuid();
gid_t getgid();



void test_note_1() {
  if (setuid(getuid()) == -1) // expected-note{{Assuming the condition is false}} \
                              // expected-note{{Taking false branch}}
    return;
  if (setuid(getuid()) == -1) // expected-note{{Call to 'setuid' found here that removes superuser privileges}} \
                              // expected-note{{Assuming the condition is false}} \
                              // expected-note{{Taking false branch}}
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                              // expected-note{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

void test_note_2() {
  if (setuid(getuid()) == -1) // expected-note{{Call to 'setuid' found here that removes superuser privileges}} \
                              // expected-note 2 {{Assuming the condition is false}} \
                              // expected-note 2 {{Taking false branch}}
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                              // expected-note{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                              // expected-note{{Assuming the condition is false}} \
                              // expected-note{{Taking false branch}}
    return;
  if (setuid(getuid()) == -1) // expected-note{{Call to 'setuid' found here that removes superuser privileges}} \
                              // expected-note{{Assuming the condition is false}} \
                              // expected-note{{Taking false branch}}
    return;
  if (setgid(getgid()) == -1) // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                              // expected-note{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    return;
}

int f_setuid() {
  return setuid(getuid()); // expected-note{{Call to 'setuid' found here that removes superuser privileges}}
}

int f_setgid() {
  return setgid(getgid()); // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                           // expected-note{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
}

void test_note_3() {
  if (f_setuid() == -1) // expected-note{{Assuming the condition is false}} \
                        // expected-note{{Calling 'f_setuid'}} \
                        // expected-note{{Returning from 'f_setuid'}} \
                        // expected-note{{Taking false branch}}
    return;
  if (f_setgid() == -1) // expected-note{{Calling 'f_setgid'}}
    return;
}

void test_note_4() {
  if (setuid(getuid()) == 0) {   // expected-note{{Assuming the condition is true}} \
                                 // expected-note{{Call to 'setuid' found here that removes superuser privileges}} \
                                 // expected-note{{Taking true branch}}
    if (setgid(getgid()) == 0) { // expected-warning{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}} \
                                 // expected-note{{A 'setgid(getgid())' call following a 'setuid(getuid())' call is likely to fail}}
    }
  }
}
