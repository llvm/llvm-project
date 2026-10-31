// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix.Stream

#include "Inputs/system-header-simulator.h"

const int Size = 10;
const char *WBuf = "123456789";
char *RBuf;

void read_write() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fread(RBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fwrite(WBuf, 1, Size, F); // expected-warning{{Output to a stream after a previous input operation without intervening position change may cause undefined behavior}}
  fclose(F);
}

void write_read() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  size_t Ret = fwrite(WBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fread(RBuf, 1, Size, F); // expected-warning{{Input from a stream after a previous output operation without intervening position change or flush may cause undefined behavior}}
  fclose(F);
}

void write_flush_read() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  size_t Ret = fwrite(WBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  if (fflush(NULL) == 0)
    fread(RBuf, 1, Size, F); // no-warning
  else
    fread(RBuf, 1, Size, F); // expected-warning{{Input from a stream after a previous output}}
  fclose(F);
}

void write_seek_read() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  size_t Ret = fwrite(WBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fseek(F, 1, SEEK_SET);
  fread(RBuf, 1, Size, F); // no-warning
  fclose(F);
}

void read_setpos_write(const fpos_t *Pos) {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fread(RBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fsetpos(F, Pos);
  fwrite(WBuf, 1, Size, F); // no-warning
  fclose(F);
}

void read_rewind_write(const fpos_t *Pos) {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fread(RBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  rewind(F);
  fwrite(WBuf, 1, Size, F); // no-warning
  fclose(F);
}

void printf_scanf() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fprintf(F, "abcdef");
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fscanf(F, "abcdef"); // expected-warning{{Input from a stream after a previous output}}
  fclose(F);
}

void getc_putc() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fgetc(F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fputc('a', F); // expected-warning{{Output to a stream after a previous input}}
  fclose(F);
}

void printf_getc() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fprintf(F, "abcdef");
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  fgetc(F); // expected-warning{{Input from a stream after a previous output}}
  fclose(F);
}

void write_clearerr_read() {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  size_t Ret = fwrite(WBuf, 1, Size, F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  clearerr(F);
  fread(RBuf, 1, Size, F); // expected-warning{{Input from a stream after a previous output}}
  fclose(F);
}

void getc_nochange_putc(fpos_t *Pos) {
  FILE *F = fopen("file.txt", "a+");
  if (F == NULL)
    return;

  fgetc(F);
  if (ferror(F) || feof(F)) {
    fclose(F);
    return;
  }
  ftell(F);
  ftello(F);
  fgetpos(F, Pos);
  ungetc('a', F);
  fileno(F);
  fputc('a', F); // expected-warning{{Output to a stream after a previous input}}
  fclose(F);
}
