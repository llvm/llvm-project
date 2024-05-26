// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix.Stream,alpha.security.taint \
// RUN:   -analyzer-checker=debug.ExprInspection

#define EOF (-1)

extern "C" {
typedef __typeof(sizeof(int)) size_t;
typedef struct _FILE FILE;

FILE *fopen(const char *filename, const char *mode);
int fclose(FILE *stream);
size_t fread(void *buffer, size_t size, size_t count, FILE *stream);
int fgetc(FILE *stream);
void *malloc(size_t size);
}

void clang_analyzer_dump(int);
void clang_analyzer_isTainted(int);
void clang_analyzer_warnIfReached();

// A stream is only tracked by StreamChecker if it results from a call to "fopen".
// Otherwise, there is no specific modelling of "fread".
void untracked_stream(FILE *fp) {
  char c;
  if (1 == fread(&c, 1, 1, fp)) {
    char p = c; // Unknown value but not garbage and not modeled by checker.
  } else {
    char p = c; // Possibly indeterminate value but not modeled by checker.
  }
}

void fgetc_props_taint() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    int c = fgetc(fp); // c is tainted.
    if (c != EOF) {
      clang_analyzer_isTainted(c); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void fread_props_taint() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    char buffer[10];
    int c = fread(buffer, 1, 10, fp); // c is tainted.
    if (c != 10) {
      // If the read failed, then the number of bytes successfully read should be tainted.
      clang_analyzer_isTainted(c); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void read_one_byte1() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    char c;
    if (1 == fread(&c, 1, 1, fp)) {
      char p = c; // Unknown value but not garbage.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    } else {
      char p = c; // Possibly indeterminate value but not modeled by checker.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void read_one_byte2(char *buffer) {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    if (1 == fread(buffer, 1, 1, fp)) {
      char p = buffer[0]; // Unknown value but not garbage.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    } else {
      char p = buffer[0]; // Possibly indeterminate value but not modeled by checker.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void read_one_byte3(char *buffer) {
  buffer[1] = 10;
  if (FILE *fp = fopen("/home/test", "rb+")) {
    // buffer[1] is not mutated by fread and remains not tainted.
    fread(buffer, 1, 1, fp);
    char p = buffer[1];
    clang_analyzer_isTainted(p); // expected-warning{{NO}}
    clang_analyzer_dump(buffer[1]); // expected-warning{{derived_}} FIXME This should be 10.
    fclose(fp);
  }
}

void read_many_bytes(char *buffer) {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    if (42 == fread(buffer, 1, 42, fp)) {
      char p = buffer[0]; // Unknown value but not garbage.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    } else {
      char p = buffer[0]; // Possibly indeterminate value but not modeled.
      clang_analyzer_isTainted(p); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void random_access_write1(int index) {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    long c[4];
    bool success = 2 == fread(c + 1, sizeof(long), 2, fp);

    switch (index) {
    case 0:
      // c[0] is not mutated by fread.
      if (success) {
        char p = c[0]; // expected-warning {{Assigned value is garbage or undefined}} We kept the first byte intact.
      } else {
        char p = c[0]; // expected-warning {{Assigned value is garbage or undefined}} We kept the first byte intact.
      }
      break;

    case 1:
      if (success) {
        // Unknown value but not garbage.
        clang_analyzer_isTainted(c[1]); // expected-warning {{YES}}
        clang_analyzer_dump(c[1]); // expected-warning {{conj_}}
      } else {
        // Possibly indeterminate value but not modeled.
        clang_analyzer_isTainted(c[1]); // expected-warning {{YES}}
        clang_analyzer_dump(c[1]); // expected-warning {{conj_}}
      }
      break;

    case 2:
      if (success) {
        long p = c[2]; // Unknown value but not garbage.
        // FIXME: Taint analysis only marks the first byte of a memory region. See getPointeeOf in GenericTaintChecker.cpp.
        clang_analyzer_isTainted(c[2]); // expected-warning {{NO}}
        clang_analyzer_dump(c[2]); // expected-warning {{conj_}}
      } else {
        // Possibly indeterminate value but not modeled.
        clang_analyzer_isTainted(c[2]); // expected-warning {{NO}} // FIXME: See above.
        clang_analyzer_dump(c[2]); // expected-warning {{conj_}}
      }
      break;

    case 3:
      // c[3] is not mutated by fread.
      if (success) {
        long p = c[3]; // expected-warning {{Assigned value is garbage or undefined}}
      } else {
        long p = c[3]; // expected-warning {{Assigned value is garbage or undefined}}
      }
      break;
    }

    fclose(fp);
  }
}

void random_access_write2(bool b) {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    int buffer[10];
    int *ptr = buffer + 2;
    if (5 == fread(ptr - 1, sizeof(int), 5, fp)) {
      if (b) {
        int p = buffer[1]; // Unknown value but not garbage.
        clang_analyzer_isTainted(p); // expected-warning {{YES}}
        clang_analyzer_dump(p); // expected-warning {{conj_}}
      } else {
        int p = buffer[0]; // expected-warning {{Assigned value is garbage or undefined}}
      }
    } else {
      int p = buffer[0]; // expected-warning {{Assigned value is garbage or undefined}}
    }
    fclose(fp);
  }
}

void random_access_write_symbolic_count(size_t count) {
  // Cover a case that used to crash (symbolic count).
  if (count > 2)
    return;

  if (FILE *fp = fopen("/home/test", "rb+")) {
    long c[4];
    fread(c + 1, sizeof(long), count, fp);

    // c[0] and c[3] are never mutated by fread, but because "count" is a symbolic value, the checker doesn't know that.
    long p = c[0];
    clang_analyzer_isTainted(p); // expected-warning {{NO}}
    clang_analyzer_dump(p); // expected-warning {{derived_}}

    p = c[3];
    clang_analyzer_isTainted(p); // expected-warning {{NO}}
    clang_analyzer_dump(p); // expected-warning {{derived_}}

    p = c[1];
    clang_analyzer_isTainted(p); // expected-warning {{YES}}
    clang_analyzer_dump(p); // expected-warning {{derived_}}

    fclose(fp);
  }
}

void dynamic_random_access_write(int startIndex) {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    long buffer[10];
    // Cannot reason about index.
    size_t res = fread(buffer + startIndex, sizeof(long), 5, fp);
    if (5 == res) {
      long p = buffer[startIndex];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    } else if (res == 4) {
      long p = buffer[startIndex];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[startIndex + 1];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[startIndex + 2];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[startIndex + 3];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[startIndex + 4];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[startIndex + 5];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[0];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    } else {
      long p = buffer[startIndex];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
      p = buffer[0];
      clang_analyzer_isTainted(p); // expected-warning {{NO}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    }
    fclose(fp);
  }
}

struct S {
  int a;
  long b;
};

void comopund_write1() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    S s; // s.a is not touched by fread.
    if (1 == fread(&s.b, sizeof(s.b), 1, fp)) {
      long p = s.b;
      clang_analyzer_isTainted(p); // expected-warning {{YES}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    } else {
      long p = s.b;
      clang_analyzer_isTainted(p); // expected-warning {{YES}}
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    }
    fclose(fp);
  }
}

void comopund_write2() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    S s; // s.a is not touched by fread.
    if (1 == fread(&s.b, sizeof(s.b), 1, fp)) {
      long p = s.a; // FIXME: This should raise an uninitialized read.
      clang_analyzer_isTainted(p); // expected-warning {{NO}} FIXME: This should be YES.
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    } else {
      long p = s.a; // FIXME: This should raise an uninitialized read.
      clang_analyzer_isTainted(p); // expected-warning {{NO}} FIXME: This should be YES.
      clang_analyzer_dump(p); // expected-warning {{conj_}}
    }
    fclose(fp);
  }
}

void var_write() {
  if (FILE *fp = fopen("/home/test", "rb+")) {
    int a, b; // 'a' is not touched by fread.
    if (1 == fread(&b, sizeof(b), 1, fp)) {
      long p = a; // expected-warning{{Assigned value is garbage or undefined}}
    } else {
      long p = a; // expected-warning{{Assigned value is garbage or undefined}}
    }
    fclose(fp);
  }
}

// When reading a lot of data, invalidating all elements is too time-consuming.
// Instead, the knowledge of the whole array is lost.
#define MaxInvalidatedElementRegion 64 // See StreamChecker::evalFreadFwrite in StreamChecker.cpp.
#define PastMaxComplexity MaxInvalidatedElementRegion + 1
void test_large_read() {
  int buffer[PastMaxComplexity + 1];
  buffer[PastMaxComplexity] = 42;
  if (FILE *fp = fopen("/home/test", "rb+")) {
    if (buffer[PastMaxComplexity] != 42) {
      clang_analyzer_warnIfReached(); // Unreachable.
    }
    if (1 == fread(buffer, sizeof(int), PastMaxComplexity, fp)) {
      if (buffer[PastMaxComplexity] != 42) {
        clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      }
    }
    fclose(fp);
  }
}

void test_small_read() {
  int buffer[10];
  buffer[5] = 42;
  if (FILE *fp = fopen("/home/test", "rb+")) {
    clang_analyzer_dump(buffer[5]); // expected-warning{{42 S32b}}
    if (1 == fread(buffer, sizeof(int), 5, fp)) {
      clang_analyzer_dump(buffer[5]); // expected-warning{{42 S32b}}
    }
    fclose(fp);
  }
}
