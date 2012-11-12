// RUN: %clang_cc1 %s -emit-llvm -o /dev/null
// PR910

struct l_struct_2E_FILE { char x; };
unsigned fwrite(signed char *, unsigned , unsigned , signed char *);
static signed char str301[39];
static void Usage(signed char *ltmp_611_6) {
  struct l_struct_2E_FILE *ltmp_6202_16;
  unsigned ltmp_6203_92;
  ltmp_6203_92 =  /*tail*/ ((unsigned  (*) (signed char *, unsigned , unsigned ,
struct l_struct_2E_FILE *))(void*)fwrite)((&(str301[0u])), 38u, 1u, ltmp_6202_16);
}
