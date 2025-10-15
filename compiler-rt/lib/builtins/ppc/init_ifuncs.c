typedef void *Ptr;
typedef struct {
  Ptr addr, toc, env;
} Descr;
typedef struct {
  Descr *desc;
  Ptr (*resolver)();
} IFUNCPair;

#define CONC2(A, B) A##B
#define CONC(A, B) CONC2(A, B)

#define IFUNC_SEC __ifunc_sec
#define IFUNC_SEC_STR "__ifunc_sec"
#define START_SEC CONC(__start_, IFUNC_SEC)
#define STOP_SEC CONC(__stop_, IFUNC_SEC)

// A zero-length entry in section "__ifunc_sec" to satisfy the START_SEC and
// STOP_SEC references in this file,  when no user code has any ifuncs.
__attribute__((section(IFUNC_SEC_STR))) static int dummy_ifunc_sec[0];

extern IFUNCPair START_SEC, STOP_SEC;

__attribute__((constructor)) void __init_ifuncs() {
  void *volatile ref = &dummy_ifunc_sec; // hack to keep dummy_ifunc_sec alive

  // hack to prevent compiler from assuming START_SEC and STOP_SEC
  // occupy different addresses.
  IFUNCPair *volatile volatile_end = &STOP_SEC;
  for (IFUNCPair *pair = &START_SEC, *end = volatile_end; pair != end; pair++) {
    // Call the resolver and copy the entire descriptor because:
    //  - the resolved function might be in another DSO, so copy the TOC address
    //  - we might be linking with objects from a language that uses the
    //    enviroment pointer, so copy it too.
    Descr *result = (Descr *)pair->resolver();
    *(pair->desc) = *result;
  }
}
