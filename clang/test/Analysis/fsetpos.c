// RUN: %clang_analyze_cc1 -analyzer-checker=unix -verify %s

#include "Inputs/system-header-simulator.h"

void set_pos_memset() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fpos_t Pos;
  memset(&Pos, 0, sizeof(Pos));
  fsetpos(F, &Pos); // expected-warning {{Position argument to 'fsetpos' (which is a constant value) was not obained from previous call to 'fgetpos' on the same file}}
  fclose(F);
}

void set_pos_fixed(FILE *F) {
  fpos_t Pos;
  *(int*)(&Pos) = 0;
  fsetpos(F, &Pos); // expected-warning {{Position argument to 'fsetpos' (which is a constant value) was not obained from previous call to 'fgetpos' on the same file}}
}

void set_pos_uninit(FILE *F) {
  fpos_t Pos;
  fsetpos(F, &Pos); // expected-warning {{Position argument to 'fsetpos' (which is an undefined value) was not obained from previous call to 'fgetpos' on the same file}}
}

void set_pos_unknown_param(FILE *F, const fpos_t *Pos) {
  fsetpos(F, Pos);
}

extern const fpos_t *get_pos(FILE *F);

void set_pos_unknown_func() {
  FILE *F = tmpfile();
  if (!F)
    return;
  const fpos_t *Pos = get_pos(F);
  fsetpos(F, Pos);
}

void set_pos_different_file() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  FILE *F2 = tmpfile();
  if (!F2) {
    fclose(F1);
    return;
  }
  fpos_t Pos1, Pos2;
  if (fgetpos(F1, &Pos1) != 0 || fgetpos(F2, &Pos2) != 0) {
    fclose(F1);
    fclose(F2);
    return;
  }
  fsetpos(F1, &Pos1);
  fsetpos(F2, &Pos1); // expected-warning {{Position argument to 'fsetpos' was not obained from previous call to 'fgetpos' on the same file}}
  fclose(F1);
  fclose(F2);
}

void set_pos_fixed_assign(FILE *F) {
  fpos_t Pos;
  *(int*)(&Pos) = 0;
  fpos_t Pos1 = Pos;
  fsetpos(F, &Pos1); // expected-warning {{Position argument to 'fsetpos' (which is a constant value) was not obained from previous call to 'fgetpos' on the same file}}
}

void set_pos_assign() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fpos_t Pos1, Pos2;
  if (fgetpos(F, &Pos1) != 0) {
    fclose(F);
    return;
  }
  Pos2 = Pos1;
  fsetpos(F, &Pos2);
  fclose(F);
}

extern void may_change_pos(FILE *, fpos_t *);

void set_pos_invalidate_1() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fpos_t Pos;
  if (fgetpos(F, &Pos) != 0) {
    fclose(F);
    return;
  }
  may_change_pos(F, &Pos);
  fsetpos(F, &Pos);
  fclose(F);
}

void set_pos_invalidate_2(FILE *F) {
  fpos_t Pos;
  may_change_pos(F, &Pos);
  fsetpos(F, &Pos);
}

void set_pos_different_file_assign() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  FILE *F2 = tmpfile();
  if (!F2) {
    fclose(F1);
    return;
  }
  fpos_t Pos1, Pos2, Pos3;
  if (fgetpos(F1, &Pos1) != 0 || fgetpos(F2, &Pos2) != 0) {
    fclose(F1);
    fclose(F2);
    return;
  }
  Pos3 = Pos1;
  fsetpos(F1, &Pos1);
  // if the 'fpos_t' type is a struct (not a simple value) this test does not pass
  fsetpos(F2, &Pos3); // expected-warning {{Position argument to 'fsetpos' was not obained from previous call to 'fgetpos' on the same file}}
  fclose(F1);
  fclose(F2);
}
