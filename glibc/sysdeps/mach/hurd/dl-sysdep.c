/* Operating system support for run-time dynamic linker.  Hurd version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* In the static library, this is all handled by dl-support.c
   or by the vanilla definitions in the rest of the C library.  */
#ifdef SHARED

#include <hurd.h>
#include <link.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <ldsodefs.h>
#include <sys/wait.h>
#include <assert.h>
#include <sysdep.h>
#include <mach/mig_support.h>
#include <mach/machine/vm_param.h>
#include "hurdstartup.h"
#include <hurd/lookup.h>
#include <hurd/auth.h>
#include <hurd/term.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <entry.h>
#include <dl-machine.h>
#include <dl-procinfo.h>

#include <dl-tunables.h>
#include <not-errno.h>
#include <not-cancel.h>

extern void __mach_init (void);

extern int _dl_argc;
extern char **_dl_argv;
extern char **_environ;

int __libc_enable_secure = 0;
rtld_hidden_data_def (__libc_enable_secure)
/* This variable contains the lowest stack address ever used.  */
void *__libc_stack_end = NULL;
rtld_hidden_data_def(__libc_stack_end)

/* TODO: Initialize.  */
void *_dl_random attribute_relro = NULL;

struct hurd_startup_data *_dl_hurd_data;

#define FMH defined(__i386__)
#if ! FMH
# define fmh()		((void)0)
# define unfmh()	((void)0)
#else
/* XXX loser kludge for vm_map kernel bug, fixed by gnumach's 0650a4ee30e3 */
#undef	ELF_MACHINE_USER_ADDRESS_MASK
#define ELF_MACHINE_USER_ADDRESS_MASK	0
static vm_address_t fmha;
static vm_size_t fmhs;
static void unfmh(void){
__vm_deallocate(__mach_task_self(),fmha,fmhs);}
static void fmh(void) {
    error_t err;int x;vm_offset_t o;mach_port_t p;
    vm_address_t a=0x08000000U,max=VM_MAX_ADDRESS;
    while (!(err=__vm_region(__mach_task_self(),&a,&fmhs,&x,&x,&x,&x,&p,&o))){
      __mach_port_deallocate(__mach_task_self(),p);
      if (a+fmhs>=0x80000000U){
	max=a; break;}
      fmha=a+=fmhs;}
    if (err) assert(err==KERN_NO_SPACE);
    if (!fmha)fmhs=0;else{
    fmhs=max-fmha;
    err = __vm_map (__mach_task_self (),
		    &fmha, fmhs, 0, 0, MACH_PORT_NULL, 0, 1,
		    VM_PROT_NONE, VM_PROT_NONE, VM_INHERIT_COPY);
    assert_perror(err);}
  }
/* XXX loser kludge for vm_map kernel bug */
#endif


ElfW(Addr)
_dl_sysdep_start (void **start_argptr,
		  void (*dl_main) (const ElfW(Phdr) *phdr, ElfW(Word) phent,
				   ElfW(Addr) *user_entry,
				   ElfW(auxv_t) *auxv))
{
  void go (intptr_t *argdata)
    {
      char **p;

      /* Cache the information in various global variables.  */
      _dl_argc = *argdata;
      _dl_argv = 1 + (char **) argdata;
      _environ = &_dl_argv[_dl_argc + 1];
      for (p = _environ; *p++;); /* Skip environ pointers and terminator.  */

      if ((void *) p == _dl_argv[0])
	{
	  static struct hurd_startup_data nodata;
	  _dl_hurd_data = &nodata;
	  nodata.user_entry = (vm_address_t) ENTRY_POINT;
	}
      else
	_dl_hurd_data = (void *) p;

      GLRO(dl_platform) = NULL; /* Default to nothing known about the platform.  */

      __libc_enable_secure = _dl_hurd_data->flags & EXEC_SECURE;

      __tunables_init (_environ);

#ifdef DL_SYSDEP_INIT
      DL_SYSDEP_INIT;
#endif

#ifdef SHARED
#ifdef DL_PLATFORM_INIT
      DL_PLATFORM_INIT;
#endif

      /* Determine the length of the platform name.  */
      if (GLRO(dl_platform) != NULL)
	GLRO(dl_platformlen) = strlen (GLRO(dl_platform));
#endif

      if (_dl_hurd_data->flags & EXEC_STACK_ARGS
	  && _dl_hurd_data->user_entry == 0)
	_dl_hurd_data->user_entry = (vm_address_t) ENTRY_POINT;

unfmh();			/* XXX */

#if 0				/* XXX make this work for real someday... */
      if (_dl_hurd_data->user_entry == (vm_address_t) ENTRY_POINT)
	/* We were invoked as a command, not as the program interpreter.
	   The generic ld.so code supports this: it will parse the args
	   as "ld.so PROGRAM [ARGS...]".  For booting the Hurd, we
	   support an additional special syntax:
	     ld.so [-LIBS...] PROGRAM [ARGS...]
	   Each LIBS word consists of "FILENAME=MEMOBJ";
	   for example "-/lib/libc.so=123" says that the contents of
	   /lib/libc.so are found in a memory object whose port name
	   in our task is 123.  */
	while (_dl_argc > 2 && _dl_argv[1][0] == '-' && _dl_argv[1][1] != '-')
	  {
	    char *lastslash, *memobjname, *p;
	    struct link_map *l;
	    mach_port_t memobj;
	    error_t err;

	    ++_dl_skip_args;
	    --_dl_argc;
	    p = _dl_argv++[1] + 1;

	    memobjname = strchr (p, '=');
	    if (! memobjname)
	      _dl_sysdep_fatal ("Bogus library spec: ", p, "\n", NULL);
	    *memobjname++ = '\0';
	    memobj = 0;
	    while (*memobjname != '\0')
	      memobj = (memobj * 10) + (*memobjname++ - '0');

	    /* Add a user reference on the memory object port, so we will
	       still have one after _dl_map_object_from_fd calls our
	       `close'.  */
	    err = __mach_port_mod_refs (__mach_task_self (), memobj,
					MACH_PORT_RIGHT_SEND, +1);
	    assert_perror (err);

	    lastslash = strrchr (p, '/');
	    l = _dl_map_object_from_fd (lastslash ? lastslash + 1 : p, NULL,
					memobj, strdup (p), 0);

	    /* Squirrel away the memory object port where it
	       can be retrieved by the program later.  */
	    l->l_info[DT_NULL] = (void *) memobj;
	  }
#endif

      /* Call elf/rtld.c's main program.  It will set everything
	 up and leave us to transfer control to USER_ENTRY.  */
      (*dl_main) ((const ElfW(Phdr) *) _dl_hurd_data->phdr,
		  _dl_hurd_data->phdrsz / sizeof (ElfW(Phdr)),
		  (ElfW(Addr) *) &_dl_hurd_data->user_entry, NULL);

      /* The call above might screw a few things up.

	 First of all, if _dl_skip_args is nonzero, we are ignoring
	 the first few arguments.  However, if we have no Hurd startup
	 data, it is the magical convention that ARGV[0] == P.  The
	 startup code in init-first.c will get confused if this is not
	 the case, so we must rearrange things to make it so.  We'll
	 overwrite the origional ARGV[0] at P with ARGV[_dl_skip_args].

	 Secondly, if we need to be secure, it removes some dangerous
	 environment variables.  If we have no Hurd startup date this
	 changes P (since that's the location after the terminating
	 NULL in the list of environment variables).  We do the same
	 thing as in the first case but make sure we recalculate P.
	 If we do have Hurd startup data, we have to move the data
	 such that it starts just after the terminating NULL in the
	 environment list.

	 We use memmove, since the locations might overlap.  */
      if (__libc_enable_secure || _dl_skip_args)
	{
	  char **newp;

	  for (newp = _environ; *newp++;);

	  if (_dl_argv[-_dl_skip_args] == (char *) p)
	    {
	      if ((char *) newp != _dl_argv[0])
		{
		  assert ((char *) newp < _dl_argv[0]);
		  _dl_argv[0] = memmove ((char *) newp, _dl_argv[0],
					 strlen (_dl_argv[0]) + 1);
		}
	    }
	  else
	    {
	      if ((void *) newp != _dl_hurd_data)
		memmove (newp, _dl_hurd_data, sizeof (*_dl_hurd_data));
	    }
	}

      {
	extern void _dl_start_user (void);
	/* Unwind the stack to ARGDATA and simulate a return from _dl_start
	   to the RTLD_START code which will run the user's entry point.  */
	RETURN_TO (argdata, &_dl_start_user, _dl_hurd_data->user_entry);
      }
    }

  /* Set up so we can do RPCs.  */
  __mach_init ();

  /* Initialize frequently used global variable.  */
  GLRO(dl_pagesize) = __getpagesize ();

fmh();				/* XXX */

  /* See hurd/hurdstartup.c; this deals with getting information
     from the exec server and slicing up the arguments.
     Then it will call `go', above.  */
  _hurd_startup (start_argptr, &go);

  LOSE;
  abort ();
}

void
_dl_sysdep_start_cleanup (void)
{
  /* Deallocate the reply port and task port rights acquired by
     __mach_init.  We are done with them now, and the user will
     reacquire them for himself when he wants them.  */
  __mig_dealloc_reply_port (MACH_PORT_NULL);
  __mach_port_deallocate (__mach_task_self (), __mach_host_self_);
  __mach_port_deallocate (__mach_task_self (), __mach_task_self_);
}

/* Minimal open/close/mmap/etc. implementation sufficient for initial loading of
   shared libraries.  These are weak definitions so that when the
   dynamic linker re-relocates itself to be user-visible (for -ldl),
   it will get the user's definition (i.e. usually libc's).

   They also need to be set in the libc and ld section of
   sysdeps/mach/hurd/Versions, to be overridable, and in libc.abilist and
   ld.abilist to be checked. */

/* This macro checks that the function does not get renamed to be hidden: we do
   need these to be overridable by libc's.  */
#define check_no_hidden(name)				\
  __typeof (name) __check_##name##_no_hidden		\
       __attribute__ ((alias (#name)))			\
       __attribute_copy__ (name);

/* Open FILE_NAME and return a Hurd I/O for it in *PORT, or return an
   error.  If STAT is non-zero, stat the file into that stat buffer.  */
static error_t
open_file (const char *file_name, int flags,
	   mach_port_t *port, struct stat64 *stat)
{
  enum retry_type doretry;
  char retryname[1024];		/* XXX string_t LOSES! */
  file_t startdir;
  error_t err;

  error_t use_init_port (int which, error_t (*operate) (file_t))
    {
      return (which < _dl_hurd_data->portarraysize
	      ? ((*operate) (_dl_hurd_data->portarray[which]))
	      : EGRATUITOUS);
    }
  file_t get_dtable_port (int fd)
    {
      if ((unsigned int) fd < _dl_hurd_data->dtablesize
	  && _dl_hurd_data->dtable[fd] != MACH_PORT_NULL)
	{
	  __mach_port_mod_refs (__mach_task_self (), _dl_hurd_data->dtable[fd],
				MACH_PORT_RIGHT_SEND, +1);
	  return _dl_hurd_data->dtable[fd];
	}
      errno = EBADF;
      return MACH_PORT_NULL;
    }

  assert (!(flags & ~(O_READ | O_CLOEXEC)));

  startdir = _dl_hurd_data->portarray[file_name[0] == '/'
				      ? INIT_PORT_CRDIR : INIT_PORT_CWDIR];

  while (file_name[0] == '/')
    file_name++;

  err = __dir_lookup (startdir, (char *)file_name, O_RDONLY, 0,
		      &doretry, retryname, port);

  if (!err)
    err = __hurd_file_name_lookup_retry (use_init_port, get_dtable_port,
					 __dir_lookup, doretry, retryname,
					 O_RDONLY, 0, port);
  if (!err && stat)
    {
      err = __io_stat (*port, stat);
      if (err)
	__mach_port_deallocate (__mach_task_self (), *port);
    }

  return err;
}

check_no_hidden(__open);
check_no_hidden (__open64);
check_no_hidden (__open_nocancel);
int weak_function
__open (const char *file_name, int mode, ...)
{
  mach_port_t port;
  error_t err = open_file (file_name, mode, &port, 0);
  if (err)
    return __hurd_fail (err);
  else
    return (int)port;
}
weak_alias (__open, __open64)
weak_alias (__open, __open_nocancel)

check_no_hidden(__close);
check_no_hidden(__close_nocancel);
int weak_function
__close (int fd)
{
  if (fd != (int) MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), (mach_port_t) fd);
  return 0;
}
weak_alias (__close, __close_nocancel)

check_no_hidden(__pread64);
check_no_hidden(__pread64_nocancel);
__ssize_t weak_function
__pread64 (int fd, void *buf, size_t nbytes, off64_t offset)
{
  error_t err;
  char *data;
  mach_msg_type_number_t nread;

  data = buf;
  nread = nbytes;
  err = __io_read ((mach_port_t) fd, &data, &nread, offset, nbytes);
  if (err)
    return __hurd_fail (err);

  if (data != buf)
    {
      memcpy (buf, data, nread);
      __vm_deallocate (__mach_task_self (), (vm_address_t) data, nread);
    }

  return nread;
}
libc_hidden_weak (__pread64)
weak_alias (__pread64, __pread64_nocancel)

check_no_hidden(__read);
check_no_hidden(__read_nocancel);
__ssize_t weak_function
__read (int fd, void *buf, size_t nbytes)
{
  return __pread64 (fd, buf, nbytes, -1);
}
libc_hidden_weak (__read)
weak_alias (__read, __read_nocancel)

check_no_hidden(__write);
check_no_hidden(__write_nocancel);
__ssize_t weak_function
__write (int fd, const void *buf, size_t nbytes)
{
  error_t err;
  mach_msg_type_number_t nwrote;

  assert (fd < _hurd_init_dtablesize);

  err = __io_write (_hurd_init_dtable[fd], buf, nbytes, -1, &nwrote);
  if (err)
    return __hurd_fail (err);

  return nwrote;
}
libc_hidden_weak (__write)
  weak_alias (__write, __write_nocancel)

/* This is only used for printing messages (see dl-misc.c).  */
check_no_hidden(__writev);
__ssize_t weak_function
__writev (int fd, const struct iovec *iov, int niov)
{
  if (fd >= _hurd_init_dtablesize)
    {
      errno = EBADF;
      return -1;
    }

  int i;
  size_t total = 0;
  for (i = 0; i < niov; ++i)
    total += iov[i].iov_len;

  if (total != 0)
    {
      char buf[total], *bufp = buf;
      error_t err;
      mach_msg_type_number_t nwrote;

      for (i = 0; i < niov; ++i)
	bufp = (memcpy (bufp, iov[i].iov_base, iov[i].iov_len)
		+ iov[i].iov_len);

      err = __io_write (_hurd_init_dtable[fd], buf, total, -1, &nwrote);
      if (err)
	return __hurd_fail (err);

      return nwrote;
    }
  return 0;
}

check_no_hidden(__libc_lseek64);
off64_t weak_function
__libc_lseek64 (int fd, off64_t offset, int whence)
{
  error_t err;

  err = __io_seek ((mach_port_t) fd, offset, whence, &offset);
  if (err)
    return __hurd_fail (err);

  return offset;
}

check_no_hidden(__mmap);
void *weak_function
__mmap (void *addr, size_t len, int prot, int flags, int fd, off_t offset)
{
  error_t err;
  vm_prot_t vmprot;
  vm_address_t mapaddr;
  mach_port_t memobj_rd, memobj_wr;

  vmprot = VM_PROT_NONE;
  if (prot & PROT_READ)
    vmprot |= VM_PROT_READ;
  if (prot & PROT_WRITE)
    vmprot |= VM_PROT_WRITE;
  if (prot & PROT_EXEC)
    vmprot |= VM_PROT_EXECUTE;

  if (flags & MAP_ANON)
    memobj_rd = MACH_PORT_NULL;
  else
    {
      assert (!(flags & MAP_SHARED));
      err = __io_map ((mach_port_t) fd, &memobj_rd, &memobj_wr);
      if (err)
	return __hurd_fail (err), MAP_FAILED;
      if (memobj_wr != MACH_PORT_NULL)
	__mach_port_deallocate (__mach_task_self (), memobj_wr);
    }

  mapaddr = (vm_address_t) addr;
  err = __vm_map (__mach_task_self (),
		  &mapaddr, (vm_size_t) len, ELF_MACHINE_USER_ADDRESS_MASK,
		  !(flags & MAP_FIXED),
		  memobj_rd,
		  (vm_offset_t) offset,
		  flags & (MAP_COPY|MAP_PRIVATE),
		  vmprot, VM_PROT_ALL,
		  (flags & MAP_SHARED) ? VM_INHERIT_SHARE : VM_INHERIT_COPY);
  if (err == KERN_NO_SPACE && (flags & MAP_FIXED))
    {
      /* XXX this is not atomic as it is in unix! */
      /* The region is already allocated; deallocate it first.  */
      err = __vm_deallocate (__mach_task_self (), mapaddr, len);
      if (! err)
	err = __vm_map (__mach_task_self (),
			&mapaddr, (vm_size_t) len,
			ELF_MACHINE_USER_ADDRESS_MASK,
			!(flags & MAP_FIXED),
			memobj_rd, (vm_offset_t) offset,
			flags & (MAP_COPY|MAP_PRIVATE),
			vmprot, VM_PROT_ALL,
			(flags & MAP_SHARED)
			? VM_INHERIT_SHARE : VM_INHERIT_COPY);
    }

  if ((flags & MAP_ANON) == 0)
    __mach_port_deallocate (__mach_task_self (), memobj_rd);

  if (err)
    return __hurd_fail (err), MAP_FAILED;
  return (void *) mapaddr;
}

check_no_hidden(__fstat64);
int weak_function
__fstat64 (int fd, struct stat64 *buf)
{
  error_t err;

  err = __io_stat ((mach_port_t) fd, buf);
  if (err)
    return __hurd_fail (err);

  return 0;
}
libc_hidden_def (__fstat64)

check_no_hidden(__stat64);
int weak_function
__stat64 (const char *file, struct stat64 *buf)
{
  error_t err;
  mach_port_t port;

  err = open_file (file, 0, &port, buf);
  if (err)
    return __hurd_fail (err);

  __mach_port_deallocate (__mach_task_self (), port);

  return 0;
}
libc_hidden_def (__stat64)

/* This function is called by the dynamic linker (rtld.c) to check
   whether debugging malloc is allowed even for SUID binaries.  This
   stub will always fail, which means that malloc-debugging is always
   disabled for SUID binaries.  */
check_no_hidden(__access);
int weak_function
__access (const char *file, int type)
{
  errno = ENOSYS;
  return -1;
}
check_no_hidden(__access_noerrno);
int weak_function
__access_noerrno (const char *file, int type)
{
  return -1;
}

check_no_hidden(__getpid);
pid_t weak_function
__getpid (void)
{
  pid_t pid, ppid;
  int orphaned;

  if (__proc_getpids (_dl_hurd_data->portarray[INIT_PORT_PROC],
		      &pid, &ppid, &orphaned))
    return -1;

  return pid;
}

/* We need this alias to satisfy references from libc_pic.a objects
   that were affected by the libc_hidden_proto declaration for __getpid.  */
strong_alias (__getpid, __GI___getpid)

/* This is called only in some strange cases trying to guess a value
   for $ORIGIN for the executable.  The dynamic linker copes with
   getcwd failing (dl-object.c), and it's too much hassle to include
   the functionality here.  (We could, it just requires duplicating or
   reusing getcwd.c's code but using our special lookup function as in
   `open', above.)  */
check_no_hidden(__getcwd);
char *weak_function
__getcwd (char *buf, size_t size)
{
  errno = ENOSYS;
  return NULL;
}

/* This is used by dl-tunables.c to strdup strings.  We can just make this a
   mere allocation.  */
check_no_hidden(__sbrk);
void *weak_function
__sbrk (intptr_t increment)
{
  vm_address_t addr;
  __vm_allocate (__mach_task_self (), &addr, increment, 1);
  return (void *) addr;
}

/* This is only used by hurdlookup for the /dev/fd/nnn magic.
 * We avoid pulling the whole libc implementation, and we can keep this hidden.  */
unsigned long int weak_function
__strtoul_internal (const char *nptr, char **endptr, int base, int group)
{
  assert (base == 0 || base == 10);
  assert (group == 0);
  return _dl_strtoul (nptr, endptr);
}

/* We need this alias to satisfy references from libc_pic.a objects
   that were affected by the libc_hidden_proto declaration for __strtoul_internal.  */
strong_alias (__strtoul_internal, __GI___strtoul_internal)
strong_alias (__strtoul_internal, __GI_____strtoul_internal)

check_no_hidden(_exit);
void weak_function attribute_hidden
_exit (int status)
{
  __proc_mark_exit (_dl_hurd_data->portarray[INIT_PORT_PROC],
		    W_EXITCODE (status, 0), 0);
  while (__task_terminate (__mach_task_self ()))
    __mach_task_self_ = (__mach_task_self) ();

  LOSE;
  abort ();
}
/* We need this alias to satisfy references from libc_pic.a objects
   that were affected by the libc_hidden_proto declaration for _exit.  */
strong_alias (_exit, __GI__exit)

/* Try to get a machine dependent instruction which will make the
   program crash.  This is used in case everything else fails.  */
#include <abort-instr.h>
#ifndef ABORT_INSTRUCTION
/* No such instruction is available.  */
# define ABORT_INSTRUCTION
#endif

check_no_hidden(abort);
void weak_function
abort (void)
{
  /* Try to abort using the system specific command.  */
  ABORT_INSTRUCTION;

  /* If the abort instruction failed, exit.  */
  _exit (127);

  /* If even this fails, make sure we never return.  */
  while (1)
    /* Try for ever and ever.  */
    ABORT_INSTRUCTION;
}

/* We need this alias to satisfy references from libc_pic.a objects
   that were affected by the libc_hidden_proto declaration for abort.  */
strong_alias (abort, __GI_abort)
strong_alias (abort, __GI___fortify_fail)
strong_alias (abort, __GI___assert_fail)
strong_alias (abort, __GI___assert_perror_fail)

/* This function is called by interruptible RPC stubs.  For initial
   dynamic linking, just use the normal mach_msg.  Since this defn is
   weak, the real defn in libc.so will override it if we are linked into
   the user program (-ldl).  */

error_t weak_function
_hurd_intr_rpc_mach_msg (mach_msg_header_t *msg,
			 mach_msg_option_t option,
			 mach_msg_size_t send_size,
			 mach_msg_size_t rcv_size,
			 mach_port_t rcv_name,
			 mach_msg_timeout_t timeout,
			 mach_port_t notify)
{
  return __mach_msg (msg, option, send_size, rcv_size, rcv_name,
		     timeout, notify);
}


void
_dl_show_auxv (void)
{
  /* There is nothing to print.  Hurd has no auxiliary vector.  */
}


void weak_function
_dl_init_first (int argc, ...)
{
  /* This no-op definition only gets used if libc is not linked in.  */
}

#endif /* SHARED */
