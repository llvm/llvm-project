// RUN: %clang_analyze_cc1 -triple=x86_64-pc-linux-gnu -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config unix.Stream:Pedantic=false -verify=nopedantic %s

// RUN: %clang_analyze_cc1 -triple=x86_64-pc-linux-gnu -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config unix.Stream:Pedantic=true -verify=pedantic %s

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);

void check_fwrite(void) {
  char *Buf = "123456789";
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  size_t Ret = fwrite(Buf, 1, 10, Fp);
  clang_analyzer_eval(Ret == 0); // nopedantic-warning {{FALSE}} \
                                 // pedantic-warning {{FALSE}} \
                                 // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fputc(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  int Ret = fputc('A', Fp);
  clang_analyzer_eval(Ret == EOF); // nopedantic-warning {{FALSE}} \
                                   // pedantic-warning {{FALSE}} \
                                   // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fputs(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  int Ret = fputs("ABC", Fp);
  clang_analyzer_eval(Ret == EOF); // nopedantic-warning {{FALSE}} \
                                   // pedantic-warning {{FALSE}} \
                                   // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fprintf(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  int Ret = fprintf(Fp, "ABC");
  clang_analyzer_eval(Ret < 0); // nopedantic-warning {{FALSE}} \
                                // pedantic-warning {{FALSE}} \
                                // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fseek(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  int Ret = fseek(Fp, 0, 0);
  clang_analyzer_eval(Ret == -1); // nopedantic-warning {{FALSE}} \
                                  // pedantic-warning {{FALSE}} \
                                  // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fseeko(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  int Ret = fseeko(Fp, 0, 0);
  clang_analyzer_eval(Ret == -1); // nopedantic-warning {{FALSE}} \
                                  // pedantic-warning {{FALSE}} \
                                  // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}

void check_fsetpos(void) {
  FILE *Fp = tmpfile();
  if (!Fp)
    return;
  fpos_t Pos;
  int Ret = fsetpos(Fp, &Pos);
  clang_analyzer_eval(Ret); // nopedantic-warning {{FALSE}} \
                            // pedantic-warning {{FALSE}} \
                            // pedantic-warning {{TRUE}}
  fputc('A', Fp); // pedantic-warning {{might be 'indeterminate'}}
  fclose(Fp);
}
