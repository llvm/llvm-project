// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.SimpleStream -verify %s

#include "Inputs/system-header-simulator-for-simple-stream.h"

void checkDoubleFClose(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
    if (!Data)
      fclose(F);
    else
      fputc(*Data, F);
    fclose(F); // expected-warning {{Closing a previously closed file stream}}
  }
}

int checkLeak(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
  }

  if (Data) // expected-warning {{Opened file is never closed; potential resource leak}}
    return *Data;
  else
    return 0;
}

void checkLeakFollowedByAssert(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
    if (!Data)
      exit(0);
    fclose(F);
  }
}

void CloseOnlyOnValidFileHandle() {
  FILE *F = fopen("myfile.txt", "w");
  if (F)
    fclose(F);
  int x = 0; // no warning
}

void leakOnEnfOfPath1(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
} // expected-warning {{Opened file is never closed; potential resource leak}}

void leakOnEnfOfPath2(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  return; // expected-warning {{Opened file is never closed; potential resource leak}}
}

FILE *leakOnEnfOfPath3(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  return F;
}

void myfclose(FILE *F);
void SymbolEscapedThroughFunctionCall() {
  FILE *F = fopen("myfile.txt", "w");
  myfclose(F);
  return; // no warning
}

FILE *GlobalF;
void SymbolEscapedThroughAssignmentToGlobal() {
  FILE *F = fopen("myfile.txt", "w");
  GlobalF = F;
  return; // no warning
}

void SymbolDoesNotEscapeThoughStringAPIs(char *Data) {
  FILE *F = fopen("myfile.txt", "w");
  fputc(*Data, F);
  return; // expected-warning {{Opened file is never closed; potential resource leak}}
}

void passConstPointer(const FILE * F);
void testPassConstPointer() {
  FILE *F = fopen("myfile.txt", "w");
  passConstPointer(F);
  return; // expected-warning {{Opened file is never closed; potential resource leak}}
}

void testPassToSystemHeaderFunctionIndirectly() {
  FileStruct fs;
  fs.p = fopen("myfile.txt", "w");
  fakeSystemHeaderCall(&fs); // invalidates fs, making fs.p unreachable
}  // no-warning
