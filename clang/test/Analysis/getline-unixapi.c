// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection -verify %s

#include "Inputs/system-header-simulator.h"
#include "Inputs/system-header-simulator-for-malloc.h"
#include "Inputs/system-header-simulator-for-valist.h"

void clang_analyzer_eval(int);
void clang_analyzer_dump_int(int);
void clang_analyzer_dump_ptr(void*);
void clang_analyzer_warnIfReached();

void test_getline_null_lineptr() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char **buffer = NULL;
  size_t n = 0;
  getline(buffer, &n, F1); // expected-warning {{Line pointer might be NULL}}
  fclose(F1);
}

void test_getline_null_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getline(&buffer, NULL, F1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}

void test_getline_null_buffer_size_gt0() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  size_t n = 8;
  getline(&buffer, &n, F1); // ok since posix 2018
  free(buffer);
  fclose(F1);
}

void test_getline_null_buffer_size_gt0_2(size_t n) {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  if (n > 0) {
    getline(&buffer, &n, F1); // ok since posix 2018
  }
  free(buffer);
  fclose(F1);
}

void test_getline_null_buffer_unknown_size(size_t n) {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;

  getline(&buffer, &n, F1);  // ok
  fclose(F1);
  free(buffer);
}

void test_getline_null_buffer_undef_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = NULL;
  size_t n;

  getline(&buffer, &n, F1); // ok since posix 2018
  fclose(F1);
  free(buffer);
}

void test_getline_buffer_size_0() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = malloc(10);
  size_t n = 0;
  if (buffer != NULL)
    getline(&buffer, &n, F1); // ok, the buffer is enough for 0 character
  fclose(F1);
  free(buffer);
}

void test_getline_buffer_bad_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = malloc(10);
  size_t n = 100;
  if (buffer != NULL)
    getline(&buffer, &n, F1); // expected-warning {{The buffer from the first argument is smaller than the size specified by the second parameter}}
  fclose(F1);
  free(buffer);
}

void test_getline_buffer_smaller_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = malloc(100);
  size_t n = 10;
  if (buffer != NULL)
    getline(&buffer, &n, F1); // ok, there is enough space for 10 characters
  fclose(F1);
  free(buffer);
}

void test_getline_buffer_undef_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  char *buffer = malloc(100);
  size_t n;
  if (buffer != NULL)
    getline(&buffer, &n, F1); // expected-warning {{The buffer from the first argument is not NULL, but the size specified by the second parameter is undefined}}
  fclose(F1);
  free(buffer);
}


void test_getline_null_buffer() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  size_t n = 0;
  ssize_t r = getline(&buffer, &n, F1);
  // getline returns -1 on failure, number of char reads on success (>= 0)
  if (r < -1) {
    clang_analyzer_warnIfReached(); // must not happen
  } else {
    // The buffer could be allocated both on failure and success
    clang_analyzer_dump_int(n);      // expected-warning {{conj_$}}
    clang_analyzer_dump_ptr(buffer); // expected-warning {{conj_$}}
  }
  free(buffer);
  fclose(F1);
}

void test_getdelim_null_size() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getdelim(&buffer, NULL, ',', F1); // expected-warning {{Size pointer might be NULL}}
  fclose(F1);
}

void test_getdelim_null_buffer_size_gt0() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  size_t n = 8;
  getdelim(&buffer, &n, ';', F1); // ok since posix 2018
  free(buffer);
  fclose(F1);
}

void test_getdelim_null_buffer_size_gt0_2(size_t n) {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  if (n > 0) {
    getdelim(&buffer, &n, ' ', F1);  // ok since posix 2018
  }
  free(buffer);
  fclose(F1);
}

void test_getdelim_null_buffer_unknown_size(size_t n) {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  getdelim(&buffer, &n, '-', F1);  // ok
  fclose(F1);
  free(buffer);
}

void test_getdelim_null_buffer() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  char *buffer = NULL;
  size_t n = 0;
  ssize_t r = getdelim(&buffer, &n, '\r', F1);
  // getdelim returns -1 on failure, number of char reads on success (>= 0)
  if (r < -1) {
    clang_analyzer_warnIfReached(); // must not happen
  }
  else {
    // The buffer could be allocated both on failure and success
    clang_analyzer_dump_int(n);      // expected-warning {{conj_$}}
    clang_analyzer_dump_ptr(buffer); // expected-warning {{conj_$}}
  }
  free(buffer);
  fclose(F1);
}

void test_getline_while() {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  while ((read = getline(&line, &len, file)) != -1) {
    printf("%s\n", line);
  }

  free(line);
  fclose(file);
}

void test_getline_return_check() {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  char *line = NULL;
  size_t len = 0;
  ssize_t r = getline(&line, &len, file);

  if (r != -1) {
    if (line[0] == '\0') {} // ok
  }
  free(line);
  fclose(file);
}

void test_getline_clear_eof() {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  size_t n = 10;
  char *buffer = malloc(n);
  ssize_t read = fread(buffer, n, 1, file);
  if (feof(file)) {
    clearerr(file);
    getline(&buffer, &n, file); // ok
  }
  fclose(file);
  free(buffer);
}

void test_getline_not_null(char **buffer, size_t *size) {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  getline(buffer, size, file);
  fclose(file);

  if (size == NULL || buffer == NULL) {
    clang_analyzer_warnIfReached(); // must not happen
  }
}

void test_getline_size_constraint(size_t size) {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  size_t old_size = size;
  char *buffer = malloc(10);
  if (buffer != NULL) {
    ssize_t r = getline(&buffer, &size, file);
    if (r >= 0) {
      // Since buffer has a size of 10, old_size must be less than or equal to 10.
      // Otherwise, there would be UB.
      clang_analyzer_eval(old_size <= 10); // expected-warning{{TRUE}}
    }
  }
  fclose(file);
  free(buffer);
}

void test_getline_negative_buffer() {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  char *buffer = NULL;
  size_t n = -1;
  getline(&buffer, &n, file); // ok since posix 2018
  free(buffer);
  fclose(file);
}

void test_getline_negative_buffer_2(char *buffer) {
  FILE *file = fopen("file.txt", "r");
  if (file == NULL) {
    return;
  }

  size_t n = -1;
  (void)getline(&buffer, &n, file); // ok
  free(buffer);
  fclose(file);
}
