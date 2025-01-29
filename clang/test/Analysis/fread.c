// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -triple x86_64-linux-gnu  \
// RUN:   -analyzer-checker=core,unix.Stream,optin.taint \
// RUN:   -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator-for-simple-stream.h"

#define EOF (-1)

void clang_analyzer_dump(int);
void clang_analyzer_dump_char(char);
void clang_analyzer_isTainted(int);
void clang_analyzer_warnIfReached(void);

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

void fgetc_props_taint(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    int c = fgetc(fp); // c is tainted.
    if (c != EOF) {
      clang_analyzer_isTainted(c); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void fread_props_taint(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    char buffer[10];
    int c = fread(buffer, 1, 10, fp); // c is tainted.
    if (c != 10) {
      // If the read failed, then the number of bytes successfully read should be tainted.
      clang_analyzer_isTainted(c); // expected-warning{{YES}}
    }
    fclose(fp);
  }
}

void read_one_byte1(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    // buffer[1] is not mutated by fread and remains not tainted.
    fread(buffer, 1, 1, fp);
    char p = buffer[1];
    clang_analyzer_isTainted(p); // expected-warning{{NO}}
    clang_analyzer_dump(buffer[1]); // expected-warning{{10 S32b}}
    fclose(fp);
  }
}

void read_many_bytes(char *buffer) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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

void random_access_read1(int index) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    long c[4];
    int success = 2 == fread(c + 1, sizeof(long), 2, fp);

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

void random_access_read2(int b) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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

void random_access_read_symbolic_count(size_t count) {
  // Cover a case that used to crash (symbolic count).
  if (count > 2)
    return;

  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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

void dynamic_random_access_read(int startIndex) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    long buffer[10];
    // Cannot reason about index.
    size_t res = fread(buffer + startIndex, sizeof(long), 5, fp);
    long *p = &buffer[startIndex];
    long v = 0;

    // If all 5 elements were successfully read, then all 5 elements should be tainted and considered initialized.
    if (5 == res) {
      // FIXME: These should be tainted.
      clang_analyzer_isTainted((v = p[0])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[1])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[2])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[3])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[4])); // expected-warning {{NO}}
      clang_analyzer_dump((v = p[0])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[1])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[2])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[3])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[4])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[5])); // expected-warning {{conj_}} FIXME: This should raise an uninit read.
    } else if (res == 4) {
      // If only the first 4 elements were successfully read,
      // then only the first 4 elements should be tainted and considered initialized.
      // FIXME: These should be tainted.
      clang_analyzer_isTainted((v = p[0])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[1])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[2])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[3])); // expected-warning {{NO}}
      clang_analyzer_dump((v = p[0])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[1])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[2])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[3])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[4])); // expected-warning {{conj_}} FIXME: This should raise an uninit read.
    } else {
      // Neither 5, or 4 elements were successfully read, so we must have read from 0 up to 3 elements.
      // FIXME: These should be tainted.
      clang_analyzer_isTainted((v = p[0])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[1])); // expected-warning {{NO}}
      clang_analyzer_isTainted((v = p[2])); // expected-warning {{NO}}
      clang_analyzer_dump((v = p[0])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[1])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[2])); // expected-warning {{conj_}} ok
      clang_analyzer_dump((v = p[3])); // expected-warning {{conj_}} FIXME: This should raise an uninit read.
    }
    fclose(fp);
  }
}

struct S {
  int a;
  long b;
};

void compound_read1(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    struct S s; // s.a is not touched by fread.
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

void compound_read2(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    struct S s; // s.a is not touched by fread.
    if (1 == fread(&s.b, sizeof(s.b), 1, fp)) {
      long p = s.a; // expected-warning {{Assigned value is garbage or undefined}}
    } else {
      long p = s.a; // expected-warning {{Assigned value is garbage or undefined}}
    }
    fclose(fp);
  }
}

void var_read(void) {
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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
void test_large_read(void) {
  int buffer[PastMaxComplexity + 1];
  buffer[PastMaxComplexity] = 42;
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
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

void test_small_read(void) {
  int buffer[10];
  buffer[5] = 42;
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    clang_analyzer_dump(buffer[5]); // expected-warning{{42 S32b}}
    if (1 == fread(buffer, sizeof(int), 5, fp)) {
      clang_analyzer_dump(buffer[5]); // expected-warning{{42 S32b}}
    }
    fclose(fp);
  }
}

void test_partial_elements_read(void) {
  clang_analyzer_dump(sizeof(int)); // expected-warning {{4 S32b}}

  int buffer[100];
  buffer[0] = 1;
  buffer[1] = 2;
  buffer[2] = 3;
  buffer[3] = 4;
  buffer[4] = 5;
  buffer[5] = 6;
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    // 3*5: 15 bytes read; which is not exactly 4 integers, but we still invalidate the first 4 ints.
    if (5 == fread(buffer + 1, 3, 5, fp)) {
      clang_analyzer_dump(buffer[0]); // expected-warning{{1 S32b}}
      clang_analyzer_dump(buffer[1]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[2]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[3]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[4]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[5]); // expected-warning{{6 S32b}}

      char *c = (char*)buffer;
      clang_analyzer_dump(c[4+12]); // expected-warning{{conj_}} 16th byte of buffer, which is the beginning of the 4th 'int' in the buffer.

      // FIXME: The store should have returned a partial binding for the 17th byte of the buffer, which is the 2nd byte of the previous int.
      // This byte should have been initialized by the 'fread' earlier. However, the Store lies to us and says it's uninitialized.
      clang_analyzer_dump(c[4+13]); // expected-warning{{1st function call argument is an uninitialized value}} should be initialized.
      clang_analyzer_dump(c[4+16]); // This should be the first byte that 'fread' leaves uninitialized. This should raise the uninit read diag.
    } else {
      clang_analyzer_dump(buffer[0]); // expected-warning{{1 S32b}} ok
      clang_analyzer_dump(buffer[1]); // expected-warning{{conj_}} ok
      clang_analyzer_dump(buffer[2]); // expected-warning{{conj_}} ok
      clang_analyzer_dump(buffer[3]); // expected-warning{{conj_}} ok
      clang_analyzer_dump(buffer[4]); // expected-warning{{conj_}} ok, but an uninit warning would be also fine.
      clang_analyzer_dump(buffer[5]); // expected-warning{{6 S32b}} ok
      clang_analyzer_dump(buffer[6]); // expected-warning{{1st function call argument is an uninitialized value}} ok
    }
    fclose(fp);
  }
}

void test_whole_elements_read(void) {
  clang_analyzer_dump(sizeof(int)); // expected-warning {{4 S32b}}

  int buffer[100];
  buffer[0] = 1;
  buffer[15] = 2;
  buffer[16] = 3;
  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    // 3*20: 60 bytes read; which is basically 15 integers.
    if (20 == fread(buffer + 1, 3, 20, fp)) {
      clang_analyzer_dump(buffer[0]);  // expected-warning{{1 S32b}}
      clang_analyzer_dump(buffer[15]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[16]); // expected-warning{{3 S32b}}
      clang_analyzer_dump(buffer[17]); // expected-warning{{1st function call argument is an uninitialized value}}
    } else {
      clang_analyzer_dump(buffer[0]);  // expected-warning{{1 S32b}}
      clang_analyzer_dump(buffer[15]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[16]); // expected-warning{{3 S32b}}
      clang_analyzer_dump(buffer[17]); // expected-warning{{1st function call argument is an uninitialized value}}
    }
    fclose(fp);
  }
}

void test_unaligned_start_read(void) {
  clang_analyzer_dump(sizeof(int)); // expected-warning {{4 S32b}}

  int buffer[100];
  buffer[0] = 3;
  buffer[1] = 4;
  buffer[2] = 5;
  char *asChar = (char*)buffer;

  FILE *fp = fopen("/home/test", "rb+");
  if (fp) {
    // We have an 'int' binding at offset 0 of value 3.
    // We read 4 bytes at byte offset: 1,2,3,4.
    if (4 == fread(asChar + 1, 1, 4, fp)) {
      clang_analyzer_dump(buffer[0]); // expected-warning{{3 S32b}} FIXME: The int binding should have been partially overwritten by the read call. This definitely should not be 3.
      clang_analyzer_dump(buffer[1]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[2]); // expected-warning{{5 S32b}}

      clang_analyzer_dump_char(asChar[0]); // expected-warning{{3 S8b}} This is technically true assuming x86 (little-endian) architecture.
      clang_analyzer_dump_char(asChar[1]); // expected-warning{{conj_}} 1
      clang_analyzer_dump_char(asChar[2]); // expected-warning{{conj_}} 2
      clang_analyzer_dump_char(asChar[3]); // expected-warning{{conj_}} 3
      clang_analyzer_dump_char(asChar[4]); // expected-warning{{conj_}} 4
      clang_analyzer_dump_char(asChar[5]); // expected-warning{{1st function call argument is an uninitialized value}}
    } else {
      clang_analyzer_dump(buffer[0]); // expected-warning{{3 S32b}} FIXME: The int binding should have been partially overwritten by the read call. This definitely should not be 3.
      clang_analyzer_dump(buffer[1]); // expected-warning{{conj_}}
      clang_analyzer_dump(buffer[2]); // expected-warning{{5 S32b}}

      clang_analyzer_dump_char(asChar[0]); // expected-warning{{3 S8b}} This is technically true assuming x86 (little-endian) architecture.
      clang_analyzer_dump_char(asChar[1]); // expected-warning{{conj_}} 1
      clang_analyzer_dump_char(asChar[2]); // expected-warning{{conj_}} 2
      clang_analyzer_dump_char(asChar[3]); // expected-warning{{conj_}} 3
      clang_analyzer_dump_char(asChar[4]); // expected-warning{{conj_}} 4
      clang_analyzer_dump_char(asChar[5]); // expected-warning{{1st function call argument is an uninitialized value}}
    }
    fclose(fp);
  }
}

void no_crash_if_count_is_negative(long l, long r, unsigned char *buffer) {
  FILE *fp = fopen("path", "r");
  if (fp) {
    if (l * r == -1) {
      fread(buffer, 1, l * r, fp); // no-crash
    }
    fclose(fp);
  }
}

void no_crash_if_size_is_negative(long l, long r, unsigned char *buffer) {
  FILE *fp = fopen("path", "r");
  if (fp) {
    if (l * r == -1) {
      fread(buffer, l * r, 1, fp); // no-crash
    }
    fclose(fp);
  }
}

void no_crash_if_size_and_count_are_negative(long l, long r, unsigned char *buffer) {
  FILE *fp = fopen("path", "r");
  if (fp) {
    if (l * r == -1) {
      fread(buffer, l * r, l * r, fp); // no-crash
    }
    fclose(fp);
  }
}
