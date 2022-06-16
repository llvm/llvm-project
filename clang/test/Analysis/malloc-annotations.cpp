// RUN: %clang_analyze_cc1 -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=alpha.core.CastSize \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:Optimistic=true %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

struct MemoryAllocator {
  void __attribute((ownership_returns(malloc))) * my_malloc(size_t);
  void __attribute((ownership_takes(malloc, 2))) my_free(void *);
  void __attribute((ownership_holds(malloc, 2))) my_hold(void *);
};

void *myglobalpointer;

struct stuff {
  void *somefield;
};

struct stuff myglobalstuff;

void af1(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  return; // expected-warning{{Potential leak of memory pointed to by}}
}

void af1_b(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
} // expected-warning{{Potential leak of memory pointed to by}}

void af1_c(MemoryAllocator &Alloc) {
  myglobalpointer = Alloc.my_malloc(12); // no-warning
}

// Test that we can pass out allocated memory via pointer-to-pointer.
void af1_e(MemoryAllocator &Alloc, void **pp) {
  *pp = Alloc.my_malloc(42); // no-warning
}

void af1_f(MemoryAllocator &Alloc, struct stuff *somestuff) {
  somestuff->somefield = Alloc.my_malloc(12); // no-warning
}

// Allocating memory for a field via multiple indirections to our arguments is OK.
void af1_g(MemoryAllocator &Alloc, struct stuff **pps) {
  *pps = (struct stuff *)Alloc.my_malloc(sizeof(struct stuff)); // no-warning
  (*pps)->somefield = Alloc.my_malloc(42); // no-warning
}

void af2(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  Alloc.my_free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void af2b(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  free(p);
  Alloc.my_free(p); // expected-warning{{Attempt to free released memory}}
}

void af2c(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  free(p);
  Alloc.my_hold(p); // expected-warning{{Attempt to free released memory}}
}

// No leak if malloc returns null.
void af2e(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  if (!p)
    return; // no-warning
  free(p); // no-warning
}

// This case inflicts a possible double-free.
void af3(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  Alloc.my_hold(p);
  free(p); // expected-warning{{Attempt to free non-owned memory}}
}

void * af4(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  Alloc.my_free(p);
  return p; // expected-warning{{Use of memory after it is freed}}
}

// This case is (possibly) ok, be conservative
void * af5(MemoryAllocator &Alloc) {
  void *p = Alloc.my_malloc(12);
  Alloc.my_hold(p);
  return p; // no-warning
}

