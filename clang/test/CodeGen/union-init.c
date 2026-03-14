// RUN: %clang_cc1 -emit-llvm < %s -o -

// A nice and complicated initialization example with unions from Python
typedef int Py_ssize_t;

typedef union _gc_head {
  struct {
    union _gc_head *gc_next;
    union _gc_head *gc_prev;
    Py_ssize_t gc_refs;
  } gc;
  long double dummy;  /* force worst-case alignment */
} PyGC_Head;

struct gc_generation {
  PyGC_Head head;
  int threshold; /* collection threshold */
  int count;     /* count of allocations or collections of younger
                    generations */
};

#define NUM_GENERATIONS 3
#define GEN_HEAD(n) (&generations[n].head)

/* linked lists of container objects */
struct gc_generation generations[NUM_GENERATIONS] = {
  /* PyGC_Head,                     threshold,      count */
  {{{GEN_HEAD(0), GEN_HEAD(0), 0}}, 700,            0},
  {{{GEN_HEAD(1), GEN_HEAD(1), 0}},  10,            0},
  {{{GEN_HEAD(2), GEN_HEAD(2), 0}},  10,            0},
};
