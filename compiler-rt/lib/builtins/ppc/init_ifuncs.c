typedef void* Ptr;
typedef struct { Ptr addr, toc, env; } Descr;
typedef struct { Descr* desc; Ptr (*resolver)(); } IFUNC_PAIR;

// A zero-length entry in section "ifunc_sec" to satisfy the __start_ifunc_sec
// and __stop_ifunc_sec references in this file, when no user code has any.
__attribute__((section("ifunc_sec"))) static int dummy_ifunc_sec[0];

extern IFUNC_PAIR __start_ifunc_sec, __stop_ifunc_sec;

__attribute__((constructor))
void __init_ifuncs() {
  void *volatile ref = &dummy_ifunc_sec; // hack to keep dummy_ifunc_sec alive

  // hack to prevent compiler from assuming __start_ifunc_sec and
  // __stop_ifunc_sec occupy different addresses.
  IFUNC_PAIR *volatile volatile_end = &__stop_ifunc_sec;
  for (IFUNC_PAIR *pair = &__start_ifunc_sec, *end = volatile_end; pair != end;
       pair++)
    pair->desc->addr = ((Descr*)(pair->resolver()))->addr;
}

