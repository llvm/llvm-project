#ifndef _NS_UNMIGRATABLE
#define _NS_UNMIGRATABLE

/* TODO(SOF-6554): Remove */
#define __PROT_MARK_AS_UNMIGRATABLE (0x100)

extern int __ns_main_app_started;

//TODO: Replace this with an explicit ns api call
static inline void __try_to_mark_as_unmigratable (void* addr)
{
  /* we do not want to depend on external headers */
  /* hence copied the x86_64 definition from stdlib/stdint.h */
  extern int __mprotect (void *__addr, size_t __len, int __prot);
  if (!__ns_main_app_started)
    return;
  /*
  * __PROT_MARK_AS_UNMIGRATABLE is intercepted by hooks, and changes the flow of mprotect completely.
  * Other protection flags are ignored.
  * Address doesn't need to be page alinged
  * Size doesn't need to be page algined
  * @see `interception_handler::mtrap_post_mprotect` in nextutils
  * @see `elrond_mem_mgr::handle_mprotect_req` in nextutils
  */
  (void) __mprotect (addr, sizeof(int), __PROT_MARK_AS_UNMIGRATABLE);
}

#endif // _NS_UNMIGRATABLE
