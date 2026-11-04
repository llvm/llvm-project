// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_CORRECT
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_1
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_2
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_3
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_4
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_5
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s -DTEST_GETLINE_GH144884

// emulator of "system-header-simulator.h" because of redefinition of 'getline' function
typedef struct _FILE FILE;
typedef __typeof(sizeof(int)) size_t;
typedef long ssize_t;
#define NULL 0

int fclose(FILE *fp);
FILE *tmpfile(void);

#ifdef TEST_CORRECT
ssize_t getline(char **lineptr, size_t *n, FILE *stream);
ssize_t getdelim(char **lineptr, size_t *n, int delimiter, FILE *stream);

void test_correct() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getline(&buffer, NULL, F1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}

void test_delim_correct() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getdelim(&buffer, NULL, ',', F1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_1
// expected-no-diagnostics
ssize_t getline(int lineptr);

void test() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  int buffer = 0;
  getline(buffer);
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_2
ssize_t getline(char **lineptr, size_t *n);

void test() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getline(&buffer, NULL); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_3
// expected-no-diagnostics
ssize_t getline(char **lineptr, size_t n, FILE *stream);

void test() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getline(&buffer, 0, F1);
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_4
ssize_t getline(char **lineptr, size_t *n, int stream);
ssize_t getdelim(char **lineptr, size_t *n, int delimiter, int stream);

void test() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getline(&buffer, NULL, 1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}

void test_delim() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getdelim(&buffer, NULL, ',', 1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_5
ssize_t getdelim(char **lineptr, size_t *n, const char* delimiter, FILE *stream);

void test_delim() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getdelim(&buffer, NULL, ",", F1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}
#endif

#ifdef TEST_GETLINE_GH144884
// expected-no-diagnostics
struct AW_string {};
void getline(int *, struct AW_string);
void top() {
  struct AW_string line;
  int getline_file_info;
  getline(&getline_file_info, line);
}
#endif
