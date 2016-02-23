// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-disable-checker=alpha.core.PointerArithm %s -analyzer-store=region -verify
// expected-no-diagnostics

typedef int bar_return_t;
typedef struct {
  unsigned char int_rep;
} Foo_record_t;
extern Foo_record_t Foo_record;
struct QuxSize {};
typedef struct QuxSize QuxSize;
typedef struct {
  Foo_record_t Foo;
  QuxSize size[0];
} __Request__SetPortalSize_t;

double __Foo_READSWAP__double(double*);

static __inline__ bar_return_t
__Beeble_check__Request__SetPortalSize_t(__Request__SetPortalSize_t *In0P) {
  if (In0P->Foo.int_rep != Foo_record.int_rep) {
    do {
      int __i__, __C__ = (2);
      for (__i__ = 0;
           __i__ < __C__;
           __i__++) do {
        *(&((double *)(&In0P->size))[__i__]) =
          __Foo_READSWAP__double(&((double *)(&In0P->size))[__i__]);
      }
      while (0);
    }
    while (0);
  }
  return 0;
}
