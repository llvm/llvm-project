// RUN: %clang_analyze_cc1 -verify -analyzer-checker=core,unix.API -analyzer-output=text %s

// Verify that the UnixAPIChecker finds the missing mode value regardless
// of the particular values of these macros, particularly O_CREAT.
#define O_RDONLY  0x2000
#define O_WRONLY  0x8000
#define O_CREAT   0x0002

extern int open(const char *path, int flags, ...);

void missing_mode_1(const char *path) {
  (void)open(path, O_CREAT); // expected-warning{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}} \
                                expected-note{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}}
}

extern int some_flag;

void missing_mode_2(const char *path) {
  int mode = O_WRONLY;
  if (some_flag) { // expected-note {{Assuming 'some_flag' is not equal to 0}} \
                     expected-note {{Taking true branch}}
    mode |= O_CREAT;
  }
  (void)open(path, mode); // expected-warning{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}} \
                             expected-note{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}}
}

void no_creat(const char* path) {
  int mode = O_RDONLY;
  (void)open(path, mode); // ok
}

void mode_is_there(const char *path) {
  int mode = O_WRONLY;
  if (some_flag) {
    mode |= O_CREAT;
  }
  (void)open(path, mode, 0770); // ok
}
