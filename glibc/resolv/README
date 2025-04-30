The resolver in the GNU C Library
*********************************

Starting with version 2.2, the resolver in the GNU C Library comes
from BIND 8.  Only a subset of the src/lib/resolv part of libbind is
included here; basically the parts that are needed to provide the
functionality present in the resolver from BIND 4.9.7 that was
included in the previous release of the GNU C Library, augmented by
the parts needed to provide thread-safety.  This means that support
for things as dynamic DNS updates and TSIG keys isn't included.  If
you need those facilities, please take a look at the full BIND
distribution.


Differences
===========

The resolver in the GNU C Library still differs from what's in BIND
8.2.3-T5B:

* The RES_DEBUG option (`options debug' in /etc/resolv.conf) has been
  disabled.

* The resolver in glibc allows underscores in domain names.

* The <resolv.h> header in glibc includes <netinet/in.h> and
  <arpa/nameser.h> to make it self-contained.

* The `res_close' function in glibc only tries to close open files
  referenced through `_res' if the RES_INIT bit is set in
  `_res.options'.  This fixes a potential security bug with programs
  that bogusly call `res_close' without initialising the resolver
  state first.  Note that the thread-safe `res_nclose' still doesn't
  check the RES_INIT bit.  By the way, you're not really supposed to
  call `res_close/res_nclose' directly.

* The resolver in glibc can connect to a nameserver over IPv6.  Just
  specify the IPv6 address in /etc/resolv.conf.  You cannot change the
  address of an IPv6 nameserver dynamically in your program though.


Using the resolver in multi-threaded code
=========================================

The traditional resolver interfaces `res_query', `res_search',
`res_mkquery', `res_send' and `res_init', used a static (global)
resolver state stored in the `_res' structure.  Therefore, these
interfaces are not thread-safe.  Therefore, BIND 8.2 introduced a set
of "new" interfaces `res_nquery', `res_nsearch', `res_nmkquery',
`res_nsend' and `res_ninit' that take a `res_state' as their first
argument, so you can use a per-thread resolver state.  In glibc, when
you link with -lpthread, such a per-thread resolver state is already
present.  It can be accessed using `_res', which has been redefined as
a macro, in a similar way to what has been done for the `errno' and
`h_errno' variables.  This per-thread resolver state is also used for
the `gethostby*' family of functions, which means that for example
`gethostbyname_r' is now fully thread-safe and re-entrant.  The
traditional resolver interfaces however, continue to use a single
resolver state and are therefore still thread-unsafe.  The resolver
state is the same resolver state that is used for the initial ("main")
thread.

This has the following consequences for existing binaries and source
code:

* Single-threaded programs will continue to work.  There should be no
  user-visible changes when you recompile them.

* Multi-threaded programs that use the traditional resolver interfaces
  in the "main" thread should continue to work, except that they no
  longer see any changes in the global resolver state caused by calls
  to, for example, `gethostbyname' in other threads.  Again there
  should be no user-visible changes when you recompile these programs.

* Multi-threaded programs that use the traditional resolver interfaces
  in more than one thread should be just as buggy as before (there are
  no problems if you use proper locking of course).  If you recompile
  these programs, manipulating the _res structure in threads other
  than the "main" thread will seem to have no effect though.

* In Multi-threaded that manipulate the _res structure, calls to
  functions like `gethostbyname' in threads other than the "main"
  thread won't be influenced by the those changes anymore.

We recommend to use the new thread-safe interfaces in new code, since
the traditional interfaces have been deprecated by the BIND folks.
For compatibility with other (older) systems you might want to
continue to use those interfaces though.


Using the resolver in C++ code
==============================

There resolver contains some hooks which will allow the user to
install some callback functions that make it possible to filter DNS
requests and responses.  Although we do not encourage you to make use
of this facility at all, C++ developers should realise that it isn't
safe to throw exceptions from such callback functions.


Source code
===========

The following files come from the BIND distribution (currently version
8.2.3-T5B):

src/include/
  arpa/nameser.h
  arpa/nameser_compat.h
  resolv.h

src/lib/resolv/
  herror.c
  res_comp.c
  res_data.c
  res_debug.c
  res_init.c
  res_mkquery.c
  res_query.c
  res_send.c

src/lib/nameser/
  ns_name.c
  ns_netint.c
  ns_parse.c
  ns_print.c
  ns_samedomain.c
  ns_ttl.c

src/lib/inet/
  inet_addr.c
  inet_net_ntop.c
  inet_net_pton.c
  inet_neta.c
  inet_ntop.c
  inet_pton.c
  nsap_addr.c

src/lib/isc/
  base64.c

Some of these files have been optimised a bit, and adaptations have
been made to make them fit in with the rest of glibc.

res_libc.c is home-brewn, although parts of it are taken from res_data.c.

res_hconf.c and res_hconf.h were contributed by David Mosberger, and
do not come from BIND.

The files gethnamaddr.c, mapv4v6addr.h and mapv4v6hostent.h are
leftovers from BIND 4.9.7.
