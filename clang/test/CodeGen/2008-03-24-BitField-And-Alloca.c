// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | not grep alloca
// RUN: %clang_cc1 -m32 -O2 -emit-llvm %s -o - | not grep {store }

enum {
 PP_C,
 PP_D,
 PP_R,
 PP_2D,
 PP_1D,
 PP_SR,
 PP_S2D,
 PP_S1D,
 PP_SC
};

enum {
 G_VP,
 G_FP,
 G_VS,
 G_GS,
 G_FS
};

enum {
 G_NONE,
 G_B,
 G_R
};

typedef union _Key {
 struct {
  unsigned int count : 2;
  unsigned int Aconst : 1;
  unsigned int Bconst : 1;
  unsigned int Cconst : 1;
  unsigned int Xused : 1;
  unsigned int Yused : 1;
  unsigned int Zused : 1;
  unsigned int Wused : 1;
  unsigned int ttype : 3;
  unsigned int scalar : 1;
  unsigned int AType : 4;
  unsigned int BType : 4;
  unsigned int CType : 4;
  unsigned int RType : 4;
  unsigned int Size : 2;
  unsigned int prec : 1;

  unsigned int ASize : 2;
  unsigned int BSize : 2;
  unsigned int CSize : 2;
  unsigned int tTex : 4;
  unsigned int proj : 1;
  unsigned int lod : 2;
  unsigned int dvts : 1;
  unsigned int uipad : 18;
 } key_io;
 struct {
  unsigned int key0;
  unsigned int key1;
 } key;
 unsigned long long lkey;
} Key;

static void foo(const Key iospec, int* ret)
{
  *ret=0;
 if(((iospec.key_io.lod == G_B) &&
  (iospec.key_io.ttype != G_VS) &&
  (iospec.key_io.ttype != G_GS) &&
  (iospec.key_io.ttype != G_FS)) ||

  (((iospec.key_io.tTex == PP_C) ||
    (iospec.key_io.tTex == PP_SC)) &&
   ((iospec.key_io.tTex == PP_SR) ||
    (iospec.key_io.tTex == PP_S2D) ||
    (iospec.key_io.tTex == PP_S1D) ||
    (iospec.key_io.tTex == PP_SC))))
  *ret=1;
}


extern int bar(unsigned long long key_token2)
{
 int ret;
 __attribute__ ((unused)) Key iospec = (Key) key_token2;
 foo(iospec, &ret);
 return ret;
}
