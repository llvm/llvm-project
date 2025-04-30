/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef	_HURD_H

#define	_HURD_H	1
#include <features.h>


/* Get types, macros, constants and function declarations
   for all Mach microkernel interaction.  */
#include <mach.h>
#include <mach/mig_errors.h>

/* Get types and constants necessary for Hurd interfaces.  */
#include <hurd/hurd_types.h>

/* Get MiG stub declarations for commonly used Hurd interfaces.  */
#include <hurd/auth.h>
#include <hurd/process.h>
#include <hurd/fs.h>
#include <hurd/io.h>

/* Get `struct hurd_port' and related definitions implementing lightweight
   user references for ports.  These are used pervasively throughout the C
   library; this is here to avoid putting it in nearly every source file.  */
#include <hurd/port.h>

#include <errno.h>
#include <bits/types/error_t.h>
#include <bits/types/sigset_t.h>

#ifndef _HURD_H_EXTERN_INLINE
#define _HURD_H_EXTERN_INLINE __extern_inline
#endif

extern int __hurd_fail (error_t err);

#ifdef __USE_EXTERN_INLINES
_HURD_H_EXTERN_INLINE int
__hurd_fail (error_t err)
{
  switch (err)
    {
    case EMACH_SEND_INVALID_DEST:
    case EMIG_SERVER_DIED:
      /* The server has disappeared!  */
      err = (error_t) EIEIO;
      break;

    case KERN_NO_SPACE:
      err = (error_t) ENOMEM;
      break;

    case KERN_INVALID_ADDRESS:
    case KERN_INVALID_ARGUMENT:
      err = (error_t) EINVAL;
      break;

    case 0:
      return 0;

    default:
      break;
    }

  errno = err;
  return -1;
}
#endif

/* Basic ports and info, initialized by startup.  */

extern int _hurd_exec_flags;	/* Flags word passed in exec_startup.  */
extern struct hurd_port *_hurd_ports;
extern unsigned int _hurd_nports;
extern mode_t _hurd_umask;
extern sigset_t _hurdsig_traced;

/* Shorthand macro for internal library code referencing _hurd_ports (see
   <hurd/port.h>).  */
/* Also see __USEPORT_CANCEL.  */

#define	__USEPORT(which, expr) \
  HURD_PORT_USE (&_hurd_ports[INIT_PORT_##which], (expr))

/* Function version of __USEPORT: calls OPERATE with a send right.  */

extern error_t _hurd_ports_use (int which, error_t (*operate) (mach_port_t));


/* Base address and size of the initial stack set up by the exec server.
   Not locked.  */

extern vm_address_t _hurd_stack_base;
extern vm_size_t _hurd_stack_size;

/* Initial file descriptor table we were passed at startup.  If we are
   using a real dtable, these are turned into that and then cleared at
   startup.  If not, these are never changed after startup.  Not locked.  */

extern mach_port_t *_hurd_init_dtable;
extern mach_msg_type_number_t _hurd_init_dtablesize;

/* Current process IDs.  */

extern pid_t _hurd_pid, _hurd_ppid, _hurd_pgrp;
extern int _hurd_orphaned;

/* This variable is incremented every time the process IDs change.  */
extern unsigned int _hurd_pids_changed_stamp;

/* Unix `data break', for brk and sbrk.
   If brk and sbrk are not used, this info will not be initialized or used.  */


/* Data break.  This is what `sbrk (0)' returns.  */

extern vm_address_t _hurd_brk;

/* End of allocated space.  This is generally `round_page (_hurd_brk)'.  */

extern vm_address_t _hurd_data_end;

/* This mutex locks _hurd_brk and _hurd_data_end.  */

extern struct mutex _hurd_brk_lock;

/* Set the data break to NEWBRK; _hurd_brk_lock must
   be held, and is released on return.  */

extern int _hurd_set_brk (vm_address_t newbrk);

#include <bits/types/FILE.h>

/* Calls to get and set basic ports.  */

extern error_t _hurd_ports_get (unsigned int which, mach_port_t *result);
extern error_t _hurd_ports_set (unsigned int which, mach_port_t newport);

extern process_t getproc (void);
extern file_t getcwdir (void), getcrdir (void);
extern auth_t getauth (void);
extern mach_port_t getcttyid (void);
extern int setproc (process_t);
extern int setcwdir (file_t), setcrdir (file_t);
extern int setcttyid (mach_port_t);

/* Does reauth with the proc server and fd io servers.  */
extern int __setauth (auth_t), setauth (auth_t);


/* Modify a port cell by looking up a directory name.
   This verifies that it is a directory and that we have search permission.  */
extern int _hurd_change_directory_port_from_name (struct hurd_port *portcell,
						  const char *name);
/* Same thing, but using an open file descriptor.
   Also verifies that it is a directory and that we have search permission.  */
extern int _hurd_change_directory_port_from_fd (struct hurd_port *portcell,
						int fd);



/* Get and set the effective UID set.  */
extern int geteuids (int __n, uid_t *__uidset);
extern int seteuids (int __n, const uid_t *__uidset);


/* Split FILE into a directory and a name within the directory.  The
   directory lookup uses the current root and working directory.  If
   successful, stores in *NAME a pointer into FILE where the name
   within directory begins and returns a port to the directory;
   otherwise sets `errno' and returns MACH_PORT_NULL.  */

extern file_t __file_name_split (const char *file, char **name);
extern file_t file_name_split (const char *file, char **name);

/* Split DIRECTORY into a parent directory and a name within the directory.
   This is the same as file_name_split, but ignores trailing slashes.  */

extern file_t __directory_name_split (const char *file, char **name);
extern file_t directory_name_split (const char *file, char **name);

/* Open a port to FILE with the given FLAGS and MODE (see <fcntl.h>).
   The file lookup uses the current root and working directory.
   Returns a port to the file if successful; otherwise sets `errno'
   and returns MACH_PORT_NULL.  */

extern file_t __file_name_lookup (const char *file, int flags, mode_t mode);
extern file_t file_name_lookup (const char *file, int flags, mode_t mode);

/* Open a port to FILE with the given FLAGS and MODE (see <fcntl.h>).  The
   file lookup uses the current root directory, but uses STARTDIR as the
   "working directory" for file relative names.  Returns a port to the file
   if successful; otherwise sets `errno' and returns MACH_PORT_NULL.  */

extern file_t __file_name_lookup_under (file_t startdir, const char *file,
					int flags, mode_t mode);
extern file_t file_name_lookup_under (file_t startdir, const char *file,
				      int flags, mode_t mode);


/* Lookup FILE_NAME and return the node opened with FLAGS & MODE
   (see hurd_file_name_lookup for details), but a simple file name (without
   any directory prefixes) will be consecutively prefixed with the pathnames
   in the `:' separated list PATH until one succeeds in a successful lookup.
   If none succeed, then the first error that wasn't ENOENT is returned, or
   ENOENT if no other errors were returned.  If PREFIXED_NAME is non-NULL,
   then if the result is looked up directly, *PREFIXED_NAME is set to NULL, and
   if it is looked up using a prefix from PATH, *PREFIXED_NAME is set to
   malloc'd storage containing the prefixed name.  */
extern file_t file_name_path_lookup (const char *file_name, const char *path,
				     int flags, mode_t mode,
				     char **prefixed_name);



/* Open a file descriptor on a port.  FLAGS are as for `open'; flags
   affected by io_set_openmodes are not changed by this.  If successful,
   this consumes a user reference for PORT (which will be deallocated on
   close).  */

extern int openport (io_t port, int flags);

/* Open a stream on a port.  MODE is as for `fopen'.
   If successful, this consumes a user reference for PORT
   (which will be deallocated on fclose).  */

extern FILE *fopenport (io_t port, const char *mode);
extern FILE *__fopenport (io_t port, const char *mode);


/* Deprecated: use _hurd_exec_paths instead.  */

extern error_t _hurd_exec (task_t task,
			   file_t file,
			   char *const argv[],
			   char *const envp[]) __attribute_deprecated__;

/* Execute a file, replacing TASK's current program image.  */

extern error_t _hurd_exec_paths (task_t task,
				 file_t file,
				 const char *path,
				 const char *abspath,
				 char *const argv[],
				 char *const envp[]);


/* Inform the proc server we have exited with STATUS, and kill the
   task thoroughly.  This function never returns, no matter what.  */

extern void _hurd_exit (int status) __attribute__ ((noreturn));


/* Initialize the library data structures from the
   ints and ports passed to us by the exec server.
   Then vm_deallocate PORTARRAY and INTARRAY.  */

extern void _hurd_init (int flags, char **argv,
			mach_port_t *portarray, size_t portarraysize,
			int *intarray, size_t intarraysize);

/* Register the process to the proc server.  */
extern void _hurd_libc_proc_init (char **argv);

/* Do startup handshaking with the proc server, and initialize library data
   structures that require proc server interaction.  This includes
   initializing signals; see _hurdsig_init in <hurd/signal.h>.  */

extern void _hurd_proc_init (char **argv,
			     const int *intarray, size_t intarraysize);


/* Return the socket server for sockaddr domain DOMAIN.  If DEAD is
   nonzero, remove the old cached port and always do a fresh lookup.

   It is assumed that a socket server will stay alive during a complex socket
   operation involving several RPCs.  But a socket server may die during
   long idle periods between socket operations.  Callers should first pass
   zero for DEAD; if the first socket RPC tried on the returned port fails
   with MACH_SEND_INVALID_DEST or MIG_SERVER_DIED (indicating the server
   went away), the caller should call _hurd_socket_server again with DEAD
   nonzero and retry the RPC on the new socket server port.  */

extern socket_t _hurd_socket_server (int domain, int dead);

/* Send a `sig_post' RPC to process number PID.  If PID is zero,
   send the message to all processes in the current process's process group.
   If PID is < -1, send SIG to all processes in process group - PID.
   SIG and REFPORT are passed along in the request message.  */

extern error_t _hurd_sig_post (pid_t pid, int sig, mach_port_t refport);
extern error_t hurd_sig_post (pid_t pid, int sig, mach_port_t refport);

/* Fetch the host privileged port and device master port from the proc
   server.  They are fetched only once and then cached in the
   variables below.  A special program that gets them from somewhere
   other than the proc server (such as a bootstrap filesystem) can set
   these variables to install the ports.  */

extern kern_return_t __get_privileged_ports (mach_port_t *host_priv_ptr,
					     device_t *device_master_ptr);
extern kern_return_t get_privileged_ports (mach_port_t *host_priv_ptr,
					   device_t *device_master_ptr);
extern mach_port_t _hurd_host_priv, _hurd_device_master;

/* Return the PID of the task whose control port is TASK.
   On error, sets `errno' and returns -1.  */

extern pid_t __task2pid (task_t task), task2pid (task_t task);

/* Return the task control port of process PID.
   On error, sets `errno' and returns MACH_PORT_NULL.  */

extern task_t __pid2task (pid_t pid), pid2task (pid_t pid);

/* Return the current thread's thread port.  This is a cheap operation (no
   system call), but it relies on Hurd signal state being set up.  */
extern thread_t hurd_thread_self (void);


/* Cancel pending operations on THREAD.  If it is doing an interruptible RPC,
   that RPC will now return EINTR; otherwise, the "cancelled" flag will be
   set, causing the next `hurd_check_cancel' call to return nonzero or the
   next interruptible RPC to return EINTR (whichever is called first).  */
extern error_t hurd_thread_cancel (thread_t thread);

/* Test and clear the calling thread's "cancelled" flag.  */
extern int hurd_check_cancel (void);


/* Return the io server port for file descriptor FD.
   This adds a Mach user reference to the returned port.
   On error, sets `errno' and returns MACH_PORT_NULL.  */

extern io_t __getdport (int fd), getdport (int fd);


#include <stdarg.h>

/* Write formatted output to PORT, a Mach port supporting the i/o protocol,
   according to the format string FORMAT, using the argument list in ARG.  */
int vpprintf (io_t port, const char *format, va_list arg);


#endif	/* hurd.h */
