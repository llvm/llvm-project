#ifndef _SEARCH_H
#include <misc/search.h>

#ifndef _ISOMAC
extern __typeof (hcreate_r) __hcreate_r;
libc_hidden_proto (__hcreate_r)
extern __typeof (hdestroy_r) __hdestroy_r;
libc_hidden_proto (__hdestroy_r)
extern __typeof (hsearch_r) __hsearch_r;
libc_hidden_proto (__hsearch_r)
libc_hidden_proto (lfind)

/* Now define the internal interfaces.  */
extern void __hdestroy (void);
extern void *__tsearch (const void *__key, void **__rootp,
			__compar_fn_t compar);
libc_hidden_proto (__tsearch)
extern void *__tfind (const void *__key, void *const *__rootp,
		      __compar_fn_t compar);
libc_hidden_proto (__tfind)
extern void *__tdelete (const void *__key, void **__rootp,
			__compar_fn_t compar);
libc_hidden_proto (__tdelete)
extern void __twalk (const void *__root, __action_fn_t action);
libc_hidden_proto (__twalk)
extern __typeof__ (twalk_r) __twalk_r;
libc_hidden_proto (__twalk_r)
extern void __tdestroy (void *__root, __free_fn_t freefct);
libc_hidden_proto (__tdestroy)
#endif
#endif
