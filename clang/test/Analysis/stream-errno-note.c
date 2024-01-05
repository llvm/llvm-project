// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.Stream \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-output text -verify %s

#include "Inputs/system-header-simulator.h"
#include "Inputs/errno_func.h"

void check_fopen(void) {
  FILE *F = fopen("xxx", "r");
  // expected-note@-1{{Assuming that 'fopen' is successful; 'errno' becomes undefined after the call}}
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [unix.Errno]}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  fclose(F);
}

void check_tmpfile(void) {
  FILE *F = tmpfile();
  // expected-note@-1{{Assuming that 'tmpfile' is successful; 'errno' becomes undefined after the call}}
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [unix.Errno]}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  fclose(F);
}

void check_freopen(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  F = freopen("xxx", "w", F);
  // expected-note@-1{{Assuming that 'freopen' is successful; 'errno' becomes undefined after the call}}
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  fclose(F);
}

void check_fclose(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  (void)fclose(F);
  // expected-note@-1{{Assuming that 'fclose' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
}

void check_fread(void) {
  char Buf[10];
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  (void)fread(Buf, 1, 10, F);
  // expected-note@-1{{Assuming that 'fread' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}

void check_fread_size0(void) {
  char Buf[10];
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  fread(Buf, 0, 1, F);
  // expected-note@-1{{Assuming that argument 'size' to 'fread' is 0; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
}

void check_fread_nmemb0(void) {
  char Buf[10];
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  fread(Buf, 1, 0, F);
  // expected-note@-1{{Assuming that 'fread' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
}

void check_fwrite(void) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  int R = fwrite(Buf, 1, 10, F);
  // expected-note@-1{{Assuming that 'fwrite' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}

void check_fseek(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  (void)fseek(F, 11, SEEK_SET);
  // expected-note@-1{{Assuming that 'fseek' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}

void check_rewind_errnocheck(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  errno = 0;
  rewind(F); // expected-note{{After calling 'rewind' reading 'errno' is required to find out if the call has failed}}
  fclose(F); // expected-warning{{Value of 'errno' was not checked and may be overwritten by function 'fclose' [unix.Errno]}}
  // expected-note@-1{{Value of 'errno' was not checked and may be overwritten by function 'fclose'}}
}

void check_fileno(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  fileno(F);
  // expected-note@-1{{Assuming that 'fileno' is successful; 'errno' becomes undefined after the call}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}

void check_fwrite_zeroarg(size_t Siz) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  errno = 0;
  int R = fwrite(Buf, Siz, 1, F);
  // expected-note@-1{{Assuming that argument 'size' to 'fwrite' is 0; 'errno' becomes undefined after the call}}
  // expected-note@+2{{'R' is <= 0}}
  // expected-note@+1{{Taking true branch}}
  if (R <= 0) {
    if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
    // expected-note@-1{{An undefined value may be read from 'errno'}}
  }
  (void)fclose(F);
}
