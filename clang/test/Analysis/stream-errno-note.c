// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.Stream \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-output text -verify %s

#include "Inputs/system-header-simulator.h"
#include "Inputs/errno_func.h"

void check_fopen(void) {
  FILE *F = fopen("xxx", "r");
  // expected-note@-1{{Assuming that function 'fopen' is successful, in this case the value 'errno' may be undefined after the call and should not be used}}
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [alpha.unix.Errno]}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  fclose(F);
}

void check_tmpfile(void) {
  FILE *F = tmpfile();
  // expected-note@-1{{Assuming that function 'tmpfile' is successful, in this case the value 'errno' may be undefined after the call and should not be used}}
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [alpha.unix.Errno]}}
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
  // expected-note@-1{{Assuming that function 'freopen' is successful}}
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
  // expected-note@-1{{Assuming that function 'fclose' is successful}}
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
  // expected-note@-1{{Assuming that function 'fread' is successful}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}

void check_fwrite(void) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  int R = fwrite(Buf, 1, 10, F);
  // expected-note@-1{{Assuming that function 'fwrite' is successful}}
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
  // expected-note@-1{{Assuming that function 'fseek' is successful}}
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
  rewind(F); // expected-note{{Function 'rewind' indicates failure only by setting of 'errno'}}
  fclose(F); // expected-warning{{Value of 'errno' was not checked and may be overwritten by function 'fclose' [alpha.unix.Errno]}}
  // expected-note@-1{{Value of 'errno' was not checked and may be overwritten by function 'fclose'}}
}

void check_fileno(void) {
  FILE *F = tmpfile();
  // expected-note@+2{{'F' is non-null}}
  // expected-note@+1{{Taking false branch}}
  if (!F)
    return;
  fileno(F);
  // expected-note@-1{{Assuming that function 'fileno' is successful}}
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
  // expected-note@-1{{An undefined value may be read from 'errno'}}
  (void)fclose(F);
}
