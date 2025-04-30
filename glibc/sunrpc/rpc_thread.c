#include <stdio.h>
#include <rpc/rpc.h>
#include <assert.h>

#include <libc-lock.h>
#include <libc-tsd.h>
#include <shlib-compat.h>
#include <libc-symbols.h>


/* Variable used in non-threaded applications or for the first thread.  */
static struct rpc_thread_variables __libc_tsd_RPC_VARS_mem;
static __thread struct rpc_thread_variables *thread_rpc_vars
        attribute_tls_model_ie;

/*
 * Task-variable destructor
 */
void
__rpc_thread_destroy (void)
{
	struct rpc_thread_variables *tvp = thread_rpc_vars;

	if (tvp != NULL) {
		__rpc_thread_svc_cleanup ();
		__rpc_thread_clnt_cleanup ();
		__rpc_thread_key_cleanup ();
		free (tvp->clnt_perr_buf_s);
		free (tvp->clntraw_private_s);
		free (tvp->svcraw_private_s);
		free (tvp->authdes_cache_s);
		free (tvp->authdes_lru_s);
		free (tvp->svc_xports_s);
		free (tvp->svc_pollfd_s);
		if (tvp != &__libc_tsd_RPC_VARS_mem)
			free (tvp);
		thread_rpc_vars = NULL;
	}
}
text_set_element (__libc_subfreeres, __rpc_thread_destroy);

/*
 * Initialize RPC multi-threaded operation
 */
static void
rpc_thread_multi (void)
{
  thread_rpc_vars = &__libc_tsd_RPC_VARS_mem;
}


struct rpc_thread_variables *
__rpc_thread_variables (void)
{
	__libc_once_define (static, once);
	struct rpc_thread_variables *tvp = thread_rpc_vars;

	if (tvp == NULL) {
		__libc_once (once, rpc_thread_multi);
		tvp = thread_rpc_vars;
		if (tvp == NULL) {
			tvp = calloc (1, sizeof *tvp);
			if (tvp != NULL)
				thread_rpc_vars = tvp;
		}
	}
	return tvp;
}


/* Global variables If we're single-threaded, or if this is the first
   thread using the variable, use the existing global variable.  This
   provides backwards compatibility for existing applications which
   dynamically link against this code.  */
#undef svc_fdset
#undef rpc_createerr
#undef svc_pollfd
#undef svc_max_pollfd

fd_set *
__rpc_thread_svc_fdset (void)
{
	struct rpc_thread_variables *tvp;

	tvp = __rpc_thread_variables ();
	if (tvp == &__libc_tsd_RPC_VARS_mem)
		return &svc_fdset;
	return &tvp->svc_fdset_s;
}
libc_hidden_nolink_sunrpc (__rpc_thread_svc_fdset, GLIBC_2_2_3)

struct rpc_createerr *
__rpc_thread_createerr (void)
{
	struct rpc_thread_variables *tvp;

	tvp = __rpc_thread_variables ();
	if (tvp == &__libc_tsd_RPC_VARS_mem)
		return &rpc_createerr;
	return &tvp->rpc_createerr_s;
}
libc_hidden_nolink_sunrpc (__rpc_thread_createerr, GLIBC_2_2_3)

struct pollfd **
__rpc_thread_svc_pollfd (void)
{
	struct rpc_thread_variables *tvp;

	tvp = __rpc_thread_variables ();
	if (tvp == &__libc_tsd_RPC_VARS_mem)
		return &svc_pollfd;
	return &tvp->svc_pollfd_s;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (__rpc_thread_svc_pollfd)
#else
libc_hidden_nolink_sunrpc (__rpc_thread_svc_pollfd, GLIBC_2_2_3)
#endif

int *
__rpc_thread_svc_max_pollfd (void)
{
	struct rpc_thread_variables *tvp;

	tvp = __rpc_thread_variables ();
	if (tvp == &__libc_tsd_RPC_VARS_mem)
		return &svc_max_pollfd;
	return &tvp->svc_max_pollfd_s;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (__rpc_thread_svc_max_pollfd)
#else
libc_hidden_nolink_sunrpc (__rpc_thread_svc_max_pollfd, GLIBC_2_2_3)
#endif
