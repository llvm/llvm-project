typedef void* Ptr;
typedef struct { Ptr addr, toc, env; } Descr;
typedef struct { Descr* desc; Ptr (*resolver)(); } IFUNC_PAIR;

extern IFUNC_PAIR __start_ifunc_sec, __stop_ifunc_sec;

__attribute__((constructor))
void __init_ifuncs() {
  for (IFUNC_PAIR *pair = &__start_ifunc_sec;
       pair != &__stop_ifunc_sec;
       pair++)
    pair->desc->addr = ((Descr*)(pair->resolver()))->addr;
}

