// This test fails on "x86_64-sie" buildbot and "x86_64-scei-ps4" target.
// According to @dyung, something related to the kind of standard library
// availability is causing the failure. Even though we explicitly define
// the relevant macros the check is hunting for in the invocation, the real
// parsing and preprocessor state will not have that case.
// UNSUPPORTED: target={{.*-(ps4|ps5)}}
//
// RUN: %check_clang_tidy -check-suffix=WITH-ANNEX-K            %s bugprone-unsafe-functions %t -- -- -D__STDC_LIB_EXT1__=1 -D__STDC_WANT_LIB_EXT1__=1
// RUN: %check_clang_tidy -check-suffix=WITHOUT-ANNEX-K         %s bugprone-unsafe-functions %t -- -- -U__STDC_LIB_EXT1__   -U__STDC_WANT_LIB_EXT1__
// RUN: %check_clang_tidy -check-suffix=WITHOUT-ANNEX-K         %s bugprone-unsafe-functions %t -- -- -D__STDC_LIB_EXT1__=1 -U__STDC_WANT_LIB_EXT1__
// RUN: %check_clang_tidy -check-suffix=WITHOUT-ANNEX-K         %s bugprone-unsafe-functions %t -- -- -U__STDC_LIB_EXT1__   -D__STDC_WANT_LIB_EXT1__=1
// RUN: %check_clang_tidy -check-suffix=WITH-ANNEX-K-CERT-ONLY  %s bugprone-unsafe-functions %t -- \
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.ReportMoreUnsafeFunctions: false}}" \
// RUN:                                                                                            -- -D__STDC_LIB_EXT1__=1 -D__STDC_WANT_LIB_EXT1__=1

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

char *gets(char *S);
size_t strlen(const char *S);
size_t wcslen(const wchar_t *S);

void f1(char *S) {
  gets(S);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'gets' is insecure, was deprecated and removed in C11 and C++14; 'gets_s' should be used instead [bugprone-unsafe-functions]
  // FIXME(?): On target=x86_64-scie-ps4, the above warning in the
  // "-WITH-ANNEX-K" case will still report the suggestion to use 'fgets'
  // instead of the expected 'get_s', as if "Annex K" was not available.
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-5]]:3: warning: function 'gets' is insecure, was deprecated and removed in C11 and C++14; 'gets_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-6]]:3: warning: function 'gets' is insecure, was deprecated and removed in C11 and C++14; 'fgets' should be used instead

  strlen(S);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'strlen' is not bounds-checking; 'strnlen_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'strlen' is not bounds-checking; 'strnlen_s' should be used instead
  // no-warning WITHOUT-ANNEX-K
}

void f1w(wchar_t *S) {
  wcslen(S);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'wcslen' is not bounds-checking; 'wcsnlen_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'wcslen' is not bounds-checking; 'wcsnlen_s' should be used instead
  // no-warning WITHOUT-ANNEX-K
}

struct tm;
char *asctime(const struct tm *TimePtr);

void f2(const struct tm *Time) {
  asctime(Time);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-3]]:3: warning: function 'asctime' is not bounds-checking and non-reentrant; 'strftime' should be used instead

  char *(*F1)(const struct tm *) = asctime;
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:36: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:36: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-3]]:36: warning: function 'asctime' is not bounds-checking and non-reentrant; 'strftime' should be used instead

  char *(*F2)(const struct tm *) = &asctime;
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:37: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:37: warning: function 'asctime' is not bounds-checking and non-reentrant; 'asctime_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-3]]:37: warning: function 'asctime' is not bounds-checking and non-reentrant; 'strftime' should be used instead
}

typedef void *FILE;
FILE *fopen(const char *Filename, const char *Mode);
FILE *freopen(const char *Filename, const char *Mode, FILE *Stream);
int fscanf(FILE *Stream, const char *Format, ...);
void rewind(FILE *Stream);
void setbuf(FILE *Stream, char *Buf);

void f3(char *S, FILE *F) {
  fopen(S, S);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'fopen' has no exclusive access to the opened file; 'fopen_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'fopen' has no exclusive access to the opened file; 'fopen_s' should be used instead
  // no-warning WITHOUT-ANNEX-K

  freopen(S, S, F);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'freopen' has no exclusive access to the opened file; 'freopen_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'freopen' has no exclusive access to the opened file; 'freopen_s' should be used instead
  // no-warning WITHOUT-ANNEX-K

  int I;
  fscanf(F, "%d", &I);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'fscanf' is not bounds-checking; 'fscanf_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'fscanf' is not bounds-checking; 'fscanf_s' should be used instead
  // no-warning WITHOUT-ANNEX-K

  rewind(F);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'rewind' has no error detection; 'fseek' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'rewind' has no error detection; 'fseek' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-3]]:3: warning: function 'rewind' has no error detection; 'fseek' should be used instead

  setbuf(F, S);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'setbuf' has no error detection; 'setvbuf' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'setbuf' has no error detection; 'setvbuf' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K:        :[[@LINE-3]]:3: warning: function 'setbuf' has no error detection; 'setvbuf' should be used instead
}

typedef int time_t;
char *ctime(const time_t *Timer);

void f4(const time_t *Timer) {
  ctime(Timer);
  // CHECK-MESSAGES-WITH-ANNEX-K:           :[[@LINE-1]]:3: warning: function 'ctime' is not bounds-checking and non-reentrant; 'ctime_s' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K-CERT-ONLY: :[[@LINE-2]]:3: warning: function 'ctime' is not bounds-checking and non-reentrant; 'ctime_s' should be used instead
  // no-warning WITHOUT-ANNEX-K
}

#define BUFSIZ 128
typedef int uid_t;
typedef int pid_t;
int bcmp(const void *S1, const void *S2, size_t N);
void bcopy(const void *Src, void *Dest, size_t N);
void bzero(void *S, size_t N);
int getpw(uid_t UId, char *Buf);
pid_t vfork(void);

void fOptional() {
  char Buf1[BUFSIZ] = {0};
  char Buf2[BUFSIZ] = {0};

  bcmp(Buf1, Buf2, BUFSIZ);
  // CHECK-MESSAGES-WITH-ANNEX-K:    :[[@LINE-1]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead
  // no-warning CERT-ONLY

  bcopy(Buf1, Buf2, BUFSIZ);
  // CHECK-MESSAGES-WITH-ANNEX-K:    :[[@LINE-1]]:3: warning: function 'bcopy' is deprecated; 'memcpy_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'bcopy' is deprecated; 'memcpy' should be used instead
  // no-warning CERT-ONLY

  bzero(Buf1, BUFSIZ);
  // CHECK-MESSAGES-WITH-ANNEX-K:    :[[@LINE-1]]:3: warning: function 'bzero' is deprecated; 'memset_s' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'bzero' is deprecated; 'memset' should be used instead
  // no-warning CERT-ONLY

  getpw(0, Buf1);
  // CHECK-MESSAGES-WITH-ANNEX-K:    :[[@LINE-1]]:3: warning: function 'getpw' is dangerous as it may overflow the provided buffer; 'getpwuid' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'getpw' is dangerous as it may overflow the provided buffer; 'getpwuid' should be used instead
  // no-warning CERT-ONLY

  vfork();
  // CHECK-MESSAGES-WITH-ANNEX-K:    :[[@LINE-1]]:3: warning: function 'vfork' is insecure as it can lead to denial of service situations in the parent process; 'posix_spawn' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'vfork' is insecure as it can lead to denial of service situations in the parent process; 'posix_spawn' should be used instead
  // no-warning CERT-ONLY
}

typedef int errno_t;
typedef size_t rsize_t;
errno_t asctime_s(char *S, rsize_t Maxsize, const struct tm *TimePtr);
errno_t strcat_s(char *S1, rsize_t S1Max, const char *S2);

void fUsingSafeFunctions(const struct tm *Time, FILE *F) {
  char Buf[BUFSIZ] = {0};

  // no-warning, safe function from annex K is used
  if (asctime_s(Buf, BUFSIZ, Time) != 0)
    return;

  // no-warning, safe function from annex K is used
  if (strcat_s(Buf, BUFSIZ, "something") != 0)
    return;
}
