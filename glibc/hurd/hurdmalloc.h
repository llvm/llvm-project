/* XXX this file is a temporary hack.

   All hurd-internal code which uses malloc et al includes this file so it
   will use the internal malloc routines _hurd_{malloc,realloc,free}
   instead.  The "hurd-internal" functions are the cthreads version,
   which uses vm_allocate and is thread-safe.  The normal user version
   of malloc et al is the unixoid one using sbrk.

 */

extern void *_hurd_malloc (size_t);
extern void *_hurd_realloc (void *, size_t);
extern void _hurd_free (void *);

extern void _hurd_malloc_fork_prepare (void);
extern void _hurd_malloc_fork_parent (void);
extern void _hurd_malloc_fork_child (void);

#define malloc	_hurd_malloc
#define realloc	_hurd_realloc
#define free	_hurd_free
