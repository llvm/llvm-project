// RUN: %check_clang_tidy -check-suffixes=DEFAULT -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "OnlyUndefined" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=WHITELIST -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "OnlyUndefined", \
// RUN:       "bugprone-exception-escape.AssumedNonThrowingFunctions": "fclose" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=OVERLAP -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.FunctionsThatShouldNotThrow": "cleanup", \
// RUN:       "bugprone-exception-escape.AssumedNonThrowingFunctions": "cleanup" \
// RUN:     }}' -- -fexceptions

struct FILE;

extern int fclose(FILE *);
extern int other_close(FILE *);

struct FileCloser {
  FileCloser(FILE *File) : File(File) {}

  ~FileCloser() {
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: an exception may be thrown in function '~FileCloser' which should not throw exceptions
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-8]]:12: note: frame #0: an exception of unknown type may be thrown in function 'fclose' here
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+1]]:5: note: frame #1: function '~FileCloser' calls function 'fclose' here
    fclose(File);
  }

  FILE *File;
};

struct OtherCloser {
  OtherCloser(FILE *File) : File(File) {}

  ~OtherCloser() {
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: an exception may be thrown in function '~OtherCloser' which should not throw exceptions
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-20]]:12: note: frame #0: an exception of unknown type may be thrown in function 'other_close' here
    // CHECK-MESSAGES-DEFAULT: :[[@LINE+4]]:5: note: frame #1: function '~OtherCloser' calls function 'other_close' here
    // CHECK-MESSAGES-WHITELIST: :[[@LINE-4]]:3: warning: an exception may be thrown in function '~OtherCloser' which should not throw exceptions
    // CHECK-MESSAGES-WHITELIST: :[[@LINE-23]]:12: note: frame #0: an exception of unknown type may be thrown in function 'other_close' here
    // CHECK-MESSAGES-WHITELIST: :[[@LINE+1]]:5: note: frame #1: function '~OtherCloser' calls function 'other_close' here
    other_close(File);
  }

  FILE *File;
};

void cleanup() {
  // CHECK-MESSAGES-OVERLAP: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'cleanup' which should not throw exceptions
  throw 1;
}
