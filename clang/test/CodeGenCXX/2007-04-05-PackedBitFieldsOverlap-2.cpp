// RUN: %clang_cc1 -emit-llvm %s -o -

#ifdef PACKED
#define P __attribute__((packed))
#else
#define P
#endif

struct P M_Packed {
  unsigned long sorted : 1;
  unsigned long from_array : 1;
  unsigned long mixed_encoding : 1;
  unsigned long encoding : 8;
  unsigned long count : 21;

};

struct M_Packed sM_Packed;

int testM_Packed (void) {
  struct M_Packed x;
  return (x.count != 0);
}
