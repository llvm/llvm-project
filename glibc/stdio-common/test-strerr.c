/* Test strerrorname_np and strerrordesc_np.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <errno.h>

#include <support/support.h>
#include <support/check.h>

static int
do_test (void)
{
  TEST_COMPARE_STRING (strerrordesc_np (0), "Success");
  TEST_COMPARE_STRING (strerrorname_np (0), "0");

#ifdef EPERM
  TEST_COMPARE_STRING (strerrordesc_np (EPERM), "Operation not permitted");
  TEST_COMPARE_STRING (strerrorname_np (EPERM), "EPERM");
#endif
#ifdef ENOENT
  TEST_COMPARE_STRING (strerrordesc_np (ENOENT),
		       "No such file or directory");
  TEST_COMPARE_STRING (strerrorname_np (ENOENT), "ENOENT");
#endif
#ifdef ESRCH
  TEST_COMPARE_STRING (strerrordesc_np (ESRCH), "No such process");
  TEST_COMPARE_STRING (strerrorname_np (ESRCH), "ESRCH");
#endif
#ifdef EINTR
  TEST_COMPARE_STRING (strerrordesc_np (EINTR), "Interrupted system call");
  TEST_COMPARE_STRING (strerrorname_np (EINTR), "EINTR");
#endif
#ifdef EIO
  TEST_COMPARE_STRING (strerrordesc_np (EIO), "Input/output error");
  TEST_COMPARE_STRING (strerrorname_np (EIO), "EIO");
#endif
#ifdef ENXIO
  TEST_COMPARE_STRING (strerrordesc_np (ENXIO), "No such device or address");
  TEST_COMPARE_STRING (strerrorname_np (ENXIO), "ENXIO");
#endif
#ifdef E2BIG
  TEST_COMPARE_STRING (strerrordesc_np (E2BIG), "Argument list too long");
  TEST_COMPARE_STRING (strerrorname_np (E2BIG), "E2BIG");
#endif
#ifdef ENOEXEC
  TEST_COMPARE_STRING (strerrordesc_np (ENOEXEC), "Exec format error");
  TEST_COMPARE_STRING (strerrorname_np (ENOEXEC), "ENOEXEC");
#endif
#ifdef EBADF
  TEST_COMPARE_STRING (strerrordesc_np (EBADF), "Bad file descriptor");
  TEST_COMPARE_STRING (strerrorname_np (EBADF), "EBADF");
#endif
#ifdef ECHILD
  TEST_COMPARE_STRING (strerrordesc_np (ECHILD), "No child processes");
  TEST_COMPARE_STRING (strerrorname_np (ECHILD), "ECHILD");
#endif
#ifdef EDEADLK
  TEST_COMPARE_STRING (strerrordesc_np (EDEADLK),
		       "Resource deadlock avoided");
  TEST_COMPARE_STRING (strerrorname_np (EDEADLK), "EDEADLK");
#endif
#ifdef ENOMEM
  TEST_COMPARE_STRING (strerrordesc_np (ENOMEM), "Cannot allocate memory");
  TEST_COMPARE_STRING (strerrorname_np (ENOMEM), "ENOMEM");
#endif
#ifdef EACCES
  TEST_COMPARE_STRING (strerrordesc_np (EACCES), "Permission denied");
  TEST_COMPARE_STRING (strerrorname_np (EACCES), "EACCES");
#endif
#ifdef EFAULT
  TEST_COMPARE_STRING (strerrordesc_np (EFAULT), "Bad address");
  TEST_COMPARE_STRING (strerrorname_np (EFAULT), "EFAULT");
#endif
#ifdef ENOTBLK
  TEST_COMPARE_STRING (strerrordesc_np (ENOTBLK), "Block device required");
  TEST_COMPARE_STRING (strerrorname_np (ENOTBLK), "ENOTBLK");
#endif
#ifdef EBUSY
  TEST_COMPARE_STRING (strerrordesc_np (EBUSY), "Device or resource busy");
  TEST_COMPARE_STRING (strerrorname_np (EBUSY), "EBUSY");
#endif
#ifdef EEXIST
  TEST_COMPARE_STRING (strerrordesc_np (EEXIST), "File exists");
  TEST_COMPARE_STRING (strerrorname_np (EEXIST), "EEXIST");
#endif
#ifdef EXDEV
  TEST_COMPARE_STRING (strerrordesc_np (EXDEV), "Invalid cross-device link");
  TEST_COMPARE_STRING (strerrorname_np (EXDEV), "EXDEV");
#endif
#ifdef ENODEV
  TEST_COMPARE_STRING (strerrordesc_np (ENODEV), "No such device");
  TEST_COMPARE_STRING (strerrorname_np (ENODEV), "ENODEV");
#endif
#ifdef ENOTDIR
  TEST_COMPARE_STRING (strerrordesc_np (ENOTDIR), "Not a directory");
  TEST_COMPARE_STRING (strerrorname_np (ENOTDIR), "ENOTDIR");
#endif
#ifdef EISDIR
  TEST_COMPARE_STRING (strerrordesc_np (EISDIR), "Is a directory");
  TEST_COMPARE_STRING (strerrorname_np (EISDIR), "EISDIR");
#endif
#ifdef EINVAL
  TEST_COMPARE_STRING (strerrordesc_np (EINVAL), "Invalid argument");
  TEST_COMPARE_STRING (strerrorname_np (EINVAL), "EINVAL");
#endif
#ifdef EMFILE
  TEST_COMPARE_STRING (strerrordesc_np (EMFILE), "Too many open files");
  TEST_COMPARE_STRING (strerrorname_np (EMFILE), "EMFILE");
#endif
#ifdef ENFILE
  TEST_COMPARE_STRING (strerrordesc_np (ENFILE),
		       "Too many open files in system");
  TEST_COMPARE_STRING (strerrorname_np (ENFILE), "ENFILE");
#endif
#ifdef ENOTTY
  TEST_COMPARE_STRING (strerrordesc_np (ENOTTY),
		       "Inappropriate ioctl for device");
  TEST_COMPARE_STRING (strerrorname_np (ENOTTY), "ENOTTY");
#endif
#ifdef ETXTBSY
  TEST_COMPARE_STRING (strerrordesc_np (ETXTBSY), "Text file busy");
  TEST_COMPARE_STRING (strerrorname_np (ETXTBSY), "ETXTBSY");
#endif
#ifdef EFBIG
  TEST_COMPARE_STRING (strerrordesc_np (EFBIG), "File too large");
  TEST_COMPARE_STRING (strerrorname_np (EFBIG), "EFBIG");
#endif
#ifdef ENOSPC
  TEST_COMPARE_STRING (strerrordesc_np (ENOSPC), "No space left on device");
  TEST_COMPARE_STRING (strerrorname_np (ENOSPC), "ENOSPC");
#endif
#ifdef ESPIPE
  TEST_COMPARE_STRING (strerrordesc_np (ESPIPE), "Illegal seek");
  TEST_COMPARE_STRING (strerrorname_np (ESPIPE), "ESPIPE");
#endif
#ifdef EROFS
  TEST_COMPARE_STRING (strerrordesc_np (EROFS), "Read-only file system");
  TEST_COMPARE_STRING (strerrorname_np (EROFS), "EROFS");
#endif
#ifdef EMLINK
  TEST_COMPARE_STRING (strerrordesc_np (EMLINK), "Too many links");
  TEST_COMPARE_STRING (strerrorname_np (EMLINK), "EMLINK");
#endif
#ifdef EPIPE
  TEST_COMPARE_STRING (strerrordesc_np (EPIPE), "Broken pipe");
  TEST_COMPARE_STRING (strerrorname_np (EPIPE), "EPIPE");
#endif
#ifdef EDOM
  TEST_COMPARE_STRING (strerrordesc_np (EDOM),
		       "Numerical argument out of domain");
  TEST_COMPARE_STRING (strerrorname_np (EDOM), "EDOM");
#endif
#ifdef ERANGE
  TEST_COMPARE_STRING (strerrordesc_np (ERANGE),
		       "Numerical result out of range");
  TEST_COMPARE_STRING (strerrorname_np (ERANGE), "ERANGE");
#endif
#ifdef EAGAIN
  TEST_COMPARE_STRING (strerrordesc_np (EAGAIN),
		       "Resource temporarily unavailable");
  TEST_COMPARE_STRING (strerrorname_np (EAGAIN), "EAGAIN");
#endif
#ifdef EINPROGRESS
  TEST_COMPARE_STRING (strerrordesc_np (EINPROGRESS),
		       "Operation now in progress");
  TEST_COMPARE_STRING (strerrorname_np (EINPROGRESS), "EINPROGRESS");
#endif
#ifdef EALREADY
  TEST_COMPARE_STRING (strerrordesc_np (EALREADY),
		       "Operation already in progress");
  TEST_COMPARE_STRING (strerrorname_np (EALREADY), "EALREADY");
#endif
#ifdef ENOTSOCK
  TEST_COMPARE_STRING (strerrordesc_np (ENOTSOCK),
		       "Socket operation on non-socket");
  TEST_COMPARE_STRING (strerrorname_np (ENOTSOCK), "ENOTSOCK");
#endif
#ifdef EMSGSIZE
  TEST_COMPARE_STRING (strerrordesc_np (EMSGSIZE), "Message too long");
  TEST_COMPARE_STRING (strerrorname_np (EMSGSIZE), "EMSGSIZE");
#endif
#ifdef EPROTOTYPE
  TEST_COMPARE_STRING (strerrordesc_np (EPROTOTYPE),
		       "Protocol wrong type for socket");
  TEST_COMPARE_STRING (strerrorname_np (EPROTOTYPE), "EPROTOTYPE");
#endif
#ifdef ENOPROTOOPT
  TEST_COMPARE_STRING (strerrordesc_np (ENOPROTOOPT),
		       "Protocol not available");
  TEST_COMPARE_STRING (strerrorname_np (ENOPROTOOPT), "ENOPROTOOPT");
#endif
#ifdef EPROTONOSUPPORT
  TEST_COMPARE_STRING (strerrordesc_np (EPROTONOSUPPORT),
		       "Protocol not supported");
  TEST_COMPARE_STRING (strerrorname_np (EPROTONOSUPPORT), "EPROTONOSUPPORT");
#endif
#ifdef ESOCKTNOSUPPORT
  TEST_COMPARE_STRING (strerrordesc_np (ESOCKTNOSUPPORT),
		       "Socket type not supported");
  TEST_COMPARE_STRING (strerrorname_np (ESOCKTNOSUPPORT), "ESOCKTNOSUPPORT");
#endif
#ifdef EOPNOTSUPP
  TEST_COMPARE_STRING (strerrordesc_np (EOPNOTSUPP),
		       "Operation not supported");
  TEST_COMPARE_STRING (strerrorname_np (EOPNOTSUPP), "EOPNOTSUPP");
#endif
#ifdef EPFNOSUPPORT
  TEST_COMPARE_STRING (strerrordesc_np (EPFNOSUPPORT),
		       "Protocol family not supported");
  TEST_COMPARE_STRING (strerrorname_np (EPFNOSUPPORT), "EPFNOSUPPORT");
#endif
#ifdef EAFNOSUPPORT
  TEST_COMPARE_STRING (strerrordesc_np (EAFNOSUPPORT),
		       "Address family not supported by protocol");
  TEST_COMPARE_STRING (strerrorname_np (EAFNOSUPPORT), "EAFNOSUPPORT");
#endif
#ifdef EADDRINUSE
  TEST_COMPARE_STRING (strerrordesc_np (EADDRINUSE),
		       "Address already in use");
  TEST_COMPARE_STRING (strerrorname_np (EADDRINUSE), "EADDRINUSE");
#endif
#ifdef EADDRNOTAVAIL
  TEST_COMPARE_STRING (strerrordesc_np (EADDRNOTAVAIL),
		       "Cannot assign requested address");
  TEST_COMPARE_STRING (strerrorname_np (EADDRNOTAVAIL), "EADDRNOTAVAIL");
#endif
#ifdef ENETDOWN
  TEST_COMPARE_STRING (strerrordesc_np (ENETDOWN), "Network is down");
  TEST_COMPARE_STRING (strerrorname_np (ENETDOWN), "ENETDOWN");
#endif
#ifdef ENETUNREACH
  TEST_COMPARE_STRING (strerrordesc_np (ENETUNREACH),
		       "Network is unreachable");
  TEST_COMPARE_STRING (strerrorname_np (ENETUNREACH), "ENETUNREACH");
#endif
#ifdef ENETRESET
  TEST_COMPARE_STRING (strerrordesc_np (ENETRESET),
		       "Network dropped connection on reset");
  TEST_COMPARE_STRING (strerrorname_np (ENETRESET), "ENETRESET");
#endif
#ifdef ECONNABORTED
  TEST_COMPARE_STRING (strerrordesc_np (ECONNABORTED),
		       "Software caused connection abort");
  TEST_COMPARE_STRING (strerrorname_np (ECONNABORTED), "ECONNABORTED");
#endif
#ifdef ECONNRESET
  TEST_COMPARE_STRING (strerrordesc_np (ECONNRESET),
		       "Connection reset by peer");
  TEST_COMPARE_STRING (strerrorname_np (ECONNRESET), "ECONNRESET");
#endif
#ifdef ENOBUFS
  TEST_COMPARE_STRING (strerrordesc_np (ENOBUFS),
		       "No buffer space available");
  TEST_COMPARE_STRING (strerrorname_np (ENOBUFS), "ENOBUFS");
#endif
#ifdef EISCONN
  TEST_COMPARE_STRING (strerrordesc_np (EISCONN),
		       "Transport endpoint is already connected");
  TEST_COMPARE_STRING (strerrorname_np (EISCONN), "EISCONN");
#endif
#ifdef ENOTCONN
  TEST_COMPARE_STRING (strerrordesc_np (ENOTCONN),
		       "Transport endpoint is not connected");
  TEST_COMPARE_STRING (strerrorname_np (ENOTCONN), "ENOTCONN");
#endif
#ifdef EDESTADDRREQ
  TEST_COMPARE_STRING (strerrordesc_np (EDESTADDRREQ),
		       "Destination address required");
  TEST_COMPARE_STRING (strerrorname_np (EDESTADDRREQ), "EDESTADDRREQ");
#endif
#ifdef ESHUTDOWN
  TEST_COMPARE_STRING (strerrordesc_np (ESHUTDOWN),
		       "Cannot send after transport endpoint shutdown");
  TEST_COMPARE_STRING (strerrorname_np (ESHUTDOWN), "ESHUTDOWN");
#endif
#ifdef ETOOMANYREFS
  TEST_COMPARE_STRING (strerrordesc_np (ETOOMANYREFS),
		       "Too many references: cannot splice");
  TEST_COMPARE_STRING (strerrorname_np (ETOOMANYREFS), "ETOOMANYREFS");
#endif
#ifdef ETIMEDOUT
  TEST_COMPARE_STRING (strerrordesc_np (ETIMEDOUT), "Connection timed out");
  TEST_COMPARE_STRING (strerrorname_np (ETIMEDOUT), "ETIMEDOUT");
#endif
#ifdef ECONNREFUSED
  TEST_COMPARE_STRING (strerrordesc_np (ECONNREFUSED), "Connection refused");
  TEST_COMPARE_STRING (strerrorname_np (ECONNREFUSED), "ECONNREFUSED");
#endif
#ifdef ELOOP
  TEST_COMPARE_STRING (strerrordesc_np (ELOOP),
		       "Too many levels of symbolic links");
  TEST_COMPARE_STRING (strerrorname_np (ELOOP), "ELOOP");
#endif
#ifdef ENAMETOOLONG
  TEST_COMPARE_STRING (strerrordesc_np (ENAMETOOLONG), "File name too long");
  TEST_COMPARE_STRING (strerrorname_np (ENAMETOOLONG), "ENAMETOOLONG");
#endif
#ifdef EHOSTDOWN
  TEST_COMPARE_STRING (strerrordesc_np (EHOSTDOWN), "Host is down");
  TEST_COMPARE_STRING (strerrorname_np (EHOSTDOWN), "EHOSTDOWN");
#endif
#ifdef EHOSTUNREACH
  TEST_COMPARE_STRING (strerrordesc_np (EHOSTUNREACH), "No route to host");
  TEST_COMPARE_STRING (strerrorname_np (EHOSTUNREACH), "EHOSTUNREACH");
#endif
#ifdef ENOTEMPTY
  TEST_COMPARE_STRING (strerrordesc_np (ENOTEMPTY), "Directory not empty");
  TEST_COMPARE_STRING (strerrorname_np (ENOTEMPTY), "ENOTEMPTY");
#endif
#ifdef EUSERS
  TEST_COMPARE_STRING (strerrordesc_np (EUSERS), "Too many users");
  TEST_COMPARE_STRING (strerrorname_np (EUSERS), "EUSERS");
#endif
#ifdef EDQUOT
  TEST_COMPARE_STRING (strerrordesc_np (EDQUOT), "Disk quota exceeded");
  TEST_COMPARE_STRING (strerrorname_np (EDQUOT), "EDQUOT");
#endif
#ifdef ESTALE
  TEST_COMPARE_STRING (strerrordesc_np (ESTALE), "Stale file handle");
  TEST_COMPARE_STRING (strerrorname_np (ESTALE), "ESTALE");
#endif
#ifdef EREMOTE
  TEST_COMPARE_STRING (strerrordesc_np (EREMOTE), "Object is remote");
  TEST_COMPARE_STRING (strerrorname_np (EREMOTE), "EREMOTE");
#endif
#ifdef ENOLCK
  TEST_COMPARE_STRING (strerrordesc_np (ENOLCK), "No locks available");
  TEST_COMPARE_STRING (strerrorname_np (ENOLCK), "ENOLCK");
#endif
#ifdef ENOSYS
  TEST_COMPARE_STRING (strerrordesc_np (ENOSYS), "Function not implemented");
  TEST_COMPARE_STRING (strerrorname_np (ENOSYS), "ENOSYS");
#endif
#ifdef EILSEQ
  TEST_COMPARE_STRING (strerrordesc_np (EILSEQ),
		       "Invalid or incomplete multibyte or wide character");
  TEST_COMPARE_STRING (strerrorname_np (EILSEQ), "EILSEQ");
#endif
#ifdef EBADMSG
  TEST_COMPARE_STRING (strerrordesc_np (EBADMSG), "Bad message");
  TEST_COMPARE_STRING (strerrorname_np (EBADMSG), "EBADMSG");
#endif
#ifdef EIDRM
  TEST_COMPARE_STRING (strerrordesc_np (EIDRM), "Identifier removed");
  TEST_COMPARE_STRING (strerrorname_np (EIDRM), "EIDRM");
#endif
#ifdef EMULTIHOP
  TEST_COMPARE_STRING (strerrordesc_np (EMULTIHOP), "Multihop attempted");
  TEST_COMPARE_STRING (strerrorname_np (EMULTIHOP), "EMULTIHOP");
#endif
#ifdef ENODATA
  TEST_COMPARE_STRING (strerrordesc_np (ENODATA), "No data available");
  TEST_COMPARE_STRING (strerrorname_np (ENODATA), "ENODATA");
#endif
#ifdef ENOLINK
  TEST_COMPARE_STRING (strerrordesc_np (ENOLINK), "Link has been severed");
  TEST_COMPARE_STRING (strerrorname_np (ENOLINK), "ENOLINK");
#endif
#ifdef ENOMSG
  TEST_COMPARE_STRING (strerrordesc_np (ENOMSG),
		       "No message of desired type");
  TEST_COMPARE_STRING (strerrorname_np (ENOMSG), "ENOMSG");
#endif
#ifdef ENOSR
  TEST_COMPARE_STRING (strerrordesc_np (ENOSR), "Out of streams resources");
  TEST_COMPARE_STRING (strerrorname_np (ENOSR), "ENOSR");
#endif
#ifdef ENOSTR
  TEST_COMPARE_STRING (strerrordesc_np (ENOSTR), "Device not a stream");
  TEST_COMPARE_STRING (strerrorname_np (ENOSTR), "ENOSTR");
#endif
#ifdef EOVERFLOW
  TEST_COMPARE_STRING (strerrordesc_np (EOVERFLOW),
		       "Value too large for defined data type");
  TEST_COMPARE_STRING (strerrorname_np (EOVERFLOW), "EOVERFLOW");
#endif
#ifdef EPROTO
  TEST_COMPARE_STRING (strerrordesc_np (EPROTO), "Protocol error");
  TEST_COMPARE_STRING (strerrorname_np (EPROTO), "EPROTO");
#endif
#ifdef ETIME
  TEST_COMPARE_STRING (strerrordesc_np (ETIME), "Timer expired");
  TEST_COMPARE_STRING (strerrorname_np (ETIME), "ETIME");
#endif
#ifdef ECANCELED
  TEST_COMPARE_STRING (strerrordesc_np (ECANCELED), "Operation canceled");
  TEST_COMPARE_STRING (strerrorname_np (ECANCELED), "ECANCELED");
#endif
#ifdef EOWNERDEAD
  TEST_COMPARE_STRING (strerrordesc_np (EOWNERDEAD), "Owner died");
  TEST_COMPARE_STRING (strerrorname_np (EOWNERDEAD), "EOWNERDEAD");
#endif
#ifdef ENOTRECOVERABLE
  TEST_COMPARE_STRING (strerrordesc_np (ENOTRECOVERABLE),
		       "State not recoverable");
  TEST_COMPARE_STRING (strerrorname_np (ENOTRECOVERABLE), "ENOTRECOVERABLE");
#endif
#ifdef ERESTART
  TEST_COMPARE_STRING (strerrordesc_np (ERESTART),
		       "Interrupted system call should be restarted");
  TEST_COMPARE_STRING (strerrorname_np (ERESTART), "ERESTART");
#endif
#ifdef ECHRNG
  TEST_COMPARE_STRING (strerrordesc_np (ECHRNG),
		       "Channel number out of range");
  TEST_COMPARE_STRING (strerrorname_np (ECHRNG), "ECHRNG");
#endif
#ifdef EL2NSYNC
  TEST_COMPARE_STRING (strerrordesc_np (EL2NSYNC),
		       "Level 2 not synchronized");
  TEST_COMPARE_STRING (strerrorname_np (EL2NSYNC), "EL2NSYNC");
#endif
#ifdef EL3HLT
  TEST_COMPARE_STRING (strerrordesc_np (EL3HLT), "Level 3 halted");
  TEST_COMPARE_STRING (strerrorname_np (EL3HLT), "EL3HLT");
#endif
#ifdef EL3RST
  TEST_COMPARE_STRING (strerrordesc_np (EL3RST), "Level 3 reset");
  TEST_COMPARE_STRING (strerrorname_np (EL3RST), "EL3RST");
#endif
#ifdef ELNRNG
  TEST_COMPARE_STRING (strerrordesc_np (ELNRNG), "Link number out of range");
  TEST_COMPARE_STRING (strerrorname_np (ELNRNG), "ELNRNG");
#endif
#ifdef EUNATCH
  TEST_COMPARE_STRING (strerrordesc_np (EUNATCH),
		       "Protocol driver not attached");
  TEST_COMPARE_STRING (strerrorname_np (EUNATCH), "EUNATCH");
#endif
#ifdef ENOCSI
  TEST_COMPARE_STRING (strerrordesc_np (ENOCSI),
		       "No CSI structure available");
  TEST_COMPARE_STRING (strerrorname_np (ENOCSI), "ENOCSI");
#endif
#ifdef EL2HLT
  TEST_COMPARE_STRING (strerrordesc_np (EL2HLT), "Level 2 halted");
  TEST_COMPARE_STRING (strerrorname_np (EL2HLT), "EL2HLT");
#endif
#ifdef EBADE
  TEST_COMPARE_STRING (strerrordesc_np (EBADE), "Invalid exchange");
  TEST_COMPARE_STRING (strerrorname_np (EBADE), "EBADE");
#endif
#ifdef EBADR
  TEST_COMPARE_STRING (strerrordesc_np (EBADR),
		       "Invalid request descriptor");
  TEST_COMPARE_STRING (strerrorname_np (EBADR), "EBADR");
#endif
#ifdef EXFULL
  TEST_COMPARE_STRING (strerrordesc_np (EXFULL), "Exchange full");
  TEST_COMPARE_STRING (strerrorname_np (EXFULL), "EXFULL");
#endif
#ifdef ENOANO
  TEST_COMPARE_STRING (strerrordesc_np (ENOANO), "No anode");
  TEST_COMPARE_STRING (strerrorname_np (ENOANO), "ENOANO");
#endif
#ifdef EBADRQC
  TEST_COMPARE_STRING (strerrordesc_np (EBADRQC), "Invalid request code");
  TEST_COMPARE_STRING (strerrorname_np (EBADRQC), "EBADRQC");
#endif
#ifdef EBADSLT
  TEST_COMPARE_STRING (strerrordesc_np (EBADSLT), "Invalid slot");
  TEST_COMPARE_STRING (strerrorname_np (EBADSLT), "EBADSLT");
#endif
#ifdef EBFONT
  TEST_COMPARE_STRING (strerrordesc_np (EBFONT), "Bad font file format");
  TEST_COMPARE_STRING (strerrorname_np (EBFONT), "EBFONT");
#endif
#ifdef ENONET
  TEST_COMPARE_STRING (strerrordesc_np (ENONET),
		       "Machine is not on the network");
  TEST_COMPARE_STRING (strerrorname_np (ENONET), "ENONET");
#endif
#ifdef ENOPKG
  TEST_COMPARE_STRING (strerrordesc_np (ENOPKG), "Package not installed");
  TEST_COMPARE_STRING (strerrorname_np (ENOPKG), "ENOPKG");
#endif
#ifdef EADV
  TEST_COMPARE_STRING (strerrordesc_np (EADV), "Advertise error");
  TEST_COMPARE_STRING (strerrorname_np (EADV), "EADV");
#endif
#ifdef ESRMNT
  TEST_COMPARE_STRING (strerrordesc_np (ESRMNT), "Srmount error");
  TEST_COMPARE_STRING (strerrorname_np (ESRMNT), "ESRMNT");
#endif
#ifdef ECOMM
  TEST_COMPARE_STRING (strerrordesc_np (ECOMM),
		       "Communication error on send");
  TEST_COMPARE_STRING (strerrorname_np (ECOMM), "ECOMM");
#endif
#ifdef EDOTDOT
  TEST_COMPARE_STRING (strerrordesc_np (EDOTDOT), "RFS specific error");
  TEST_COMPARE_STRING (strerrorname_np (EDOTDOT), "EDOTDOT");
#endif
#ifdef ENOTUNIQ
  TEST_COMPARE_STRING (strerrordesc_np (ENOTUNIQ),
		       "Name not unique on network");
  TEST_COMPARE_STRING (strerrorname_np (ENOTUNIQ), "ENOTUNIQ");
#endif
#ifdef EBADFD
  TEST_COMPARE_STRING (strerrordesc_np (EBADFD),
		       "File descriptor in bad state");
  TEST_COMPARE_STRING (strerrorname_np (EBADFD), "EBADFD");
#endif
#ifdef EREMCHG
  TEST_COMPARE_STRING (strerrordesc_np (EREMCHG), "Remote address changed");
  TEST_COMPARE_STRING (strerrorname_np (EREMCHG), "EREMCHG");
#endif
#ifdef ELIBACC
  TEST_COMPARE_STRING (strerrordesc_np (ELIBACC),
		       "Can not access a needed shared library");
  TEST_COMPARE_STRING (strerrorname_np (ELIBACC), "ELIBACC");
#endif
#ifdef ELIBBAD
  TEST_COMPARE_STRING (strerrordesc_np (ELIBBAD),
		       "Accessing a corrupted shared library");
  TEST_COMPARE_STRING (strerrorname_np (ELIBBAD), "ELIBBAD");
#endif
#ifdef ELIBSCN
  TEST_COMPARE_STRING (strerrordesc_np (ELIBSCN),
		       ".lib section in a.out corrupted");
  TEST_COMPARE_STRING (strerrorname_np (ELIBSCN), "ELIBSCN");
#endif
#ifdef ELIBMAX
  TEST_COMPARE_STRING (strerrordesc_np (ELIBMAX),
		       "Attempting to link in too many shared libraries");
  TEST_COMPARE_STRING (strerrorname_np (ELIBMAX), "ELIBMAX");
#endif
#ifdef ELIBEXEC
  TEST_COMPARE_STRING (strerrordesc_np (ELIBEXEC),
		       "Cannot exec a shared library directly");
  TEST_COMPARE_STRING (strerrorname_np (ELIBEXEC), "ELIBEXEC");
#endif
#ifdef ESTRPIPE
  TEST_COMPARE_STRING (strerrordesc_np (ESTRPIPE), "Streams pipe error");
  TEST_COMPARE_STRING (strerrorname_np (ESTRPIPE), "ESTRPIPE");
#endif
#ifdef EUCLEAN
  TEST_COMPARE_STRING (strerrordesc_np (EUCLEAN),
		       "Structure needs cleaning");
  TEST_COMPARE_STRING (strerrorname_np (EUCLEAN), "EUCLEAN");
#endif
#ifdef ENOTNAM
  TEST_COMPARE_STRING (strerrordesc_np (ENOTNAM),
		       "Not a XENIX named type file");
  TEST_COMPARE_STRING (strerrorname_np (ENOTNAM), "ENOTNAM");
#endif
#ifdef ENAVAIL
  TEST_COMPARE_STRING (strerrordesc_np (ENAVAIL),
		       "No XENIX semaphores available");
  TEST_COMPARE_STRING (strerrorname_np (ENAVAIL), "ENAVAIL");
#endif
#ifdef EISNAM
  TEST_COMPARE_STRING (strerrordesc_np (EISNAM), "Is a named type file");
  TEST_COMPARE_STRING (strerrorname_np (EISNAM), "EISNAM");
#endif
#ifdef EREMOTEIO
  TEST_COMPARE_STRING (strerrordesc_np (EREMOTEIO), "Remote I/O error");
  TEST_COMPARE_STRING (strerrorname_np (EREMOTEIO), "EREMOTEIO");
#endif
#ifdef ENOMEDIUM
  TEST_COMPARE_STRING (strerrordesc_np (ENOMEDIUM), "No medium found");
  TEST_COMPARE_STRING (strerrorname_np (ENOMEDIUM), "ENOMEDIUM");
#endif
#ifdef EMEDIUMTYPE
  TEST_COMPARE_STRING (strerrordesc_np (EMEDIUMTYPE), "Wrong medium type");
  TEST_COMPARE_STRING (strerrorname_np (EMEDIUMTYPE), "EMEDIUMTYPE");
#endif
#ifdef ENOKEY
  TEST_COMPARE_STRING (strerrordesc_np (ENOKEY),
		       "Required key not available");
  TEST_COMPARE_STRING (strerrorname_np (ENOKEY), "ENOKEY");
#endif
#ifdef EKEYEXPIRED
  TEST_COMPARE_STRING (strerrordesc_np (EKEYEXPIRED), "Key has expired");
  TEST_COMPARE_STRING (strerrorname_np (EKEYEXPIRED), "EKEYEXPIRED");
#endif
#ifdef EKEYREVOKED
  TEST_COMPARE_STRING (strerrordesc_np (EKEYREVOKED),
		       "Key has been revoked");
  TEST_COMPARE_STRING (strerrorname_np (EKEYREVOKED), "EKEYREVOKED");
#endif
#ifdef EKEYREJECTED
  TEST_COMPARE_STRING (strerrordesc_np (EKEYREJECTED),
		       "Key was rejected by service");
  TEST_COMPARE_STRING (strerrorname_np (EKEYREJECTED), "EKEYREJECTED");
#endif
#ifdef ERFKILL
  TEST_COMPARE_STRING (strerrordesc_np (ERFKILL),
		       "Operation not possible due to RF-kill");
  TEST_COMPARE_STRING (strerrorname_np (ERFKILL), "ERFKILL");
#endif
#ifdef EHWPOISON
  TEST_COMPARE_STRING (strerrordesc_np (EHWPOISON),
		       "Memory page has hardware error");
  TEST_COMPARE_STRING (strerrorname_np (EHWPOISON), "EHWPOISON");
#endif
#ifdef EBADRPC
  TEST_COMPARE_STRING (strerrordesc_np (EBADRPC), "RPC struct is bad");
  TEST_COMPARE_STRING (strerrorname_np (EBADRPC), "EBADRPC");
#endif
#ifdef EFTYPE
  TEST_COMPARE_STRING (strerrordesc_np (EFTYPE),
		       "Inappropriate file type or format");
  TEST_COMPARE_STRING (strerrorname_np (EFTYPE), "EFTYPE");
#endif
#ifdef EPROCUNAVAIL
  TEST_COMPARE_STRING (strerrordesc_np (EPROCUNAVAIL),
		       "RPC bad procedure for program");
  TEST_COMPARE_STRING (strerrorname_np (EPROCUNAVAIL), "EPROCUNAVAIL");
#endif
#ifdef EAUTH
  TEST_COMPARE_STRING (strerrordesc_np (EAUTH), "Authentication error");
  TEST_COMPARE_STRING (strerrorname_np (EAUTH), "EAUTH");
#endif
#ifdef EDIED
  TEST_COMPARE_STRING (strerrordesc_np (EDIED), "Translator died");
  TEST_COMPARE_STRING (strerrorname_np (EDIED), "EDIED");
#endif
#ifdef ERPCMISMATCH
  TEST_COMPARE_STRING (strerrordesc_np (ERPCMISMATCH), "RPC version wrong");
  TEST_COMPARE_STRING (strerrorname_np (ERPCMISMATCH), "ERPCMISMATCH");
#endif
#ifdef EGREGIOUS
  TEST_COMPARE_STRING (strerrordesc_np (EGREGIOUS),
		       "You really blew it this time");
  TEST_COMPARE_STRING (strerrorname_np (EGREGIOUS), "EGREGIOUS");
#endif
#ifdef EPROCLIM
  TEST_COMPARE_STRING (strerrordesc_np (EPROCLIM), "Too many processes");
  TEST_COMPARE_STRING (strerrorname_np (EPROCLIM), "EPROCLIM");
#endif
#ifdef EGRATUITOUS
  TEST_COMPARE_STRING (strerrordesc_np (EGRATUITOUS), "Gratuitous error");
  TEST_COMPARE_STRING (strerrorname_np (EGRATUITOUS), "EGRATUITOUS");
#endif
#if defined (ENOTSUP) && ENOTSUP != EOPNOTSUPP
  TEST_COMPARE_STRING (strerrordesc_np (ENOTSUP), "Not supported");
  TEST_COMPARE_STRING (strerrorname_np (ENOTSUP), "ENOTSUP");
#endif
#ifdef EPROGMISMATCH
  TEST_COMPARE_STRING (strerrordesc_np (EPROGMISMATCH),
		       "RPC program version wrong");
  TEST_COMPARE_STRING (strerrorname_np (EPROGMISMATCH), "EPROGMISMATCH");
#endif
#ifdef EBACKGROUND
  TEST_COMPARE_STRING (strerrordesc_np (EBACKGROUND),
		       "Inappropriate operation for background process");
  TEST_COMPARE_STRING (strerrorname_np (EBACKGROUND), "EBACKGROUND");
#endif
#ifdef EIEIO
  TEST_COMPARE_STRING (strerrordesc_np (EIEIO), "Computer bought the farm");
  TEST_COMPARE_STRING (strerrorname_np (EIEIO), "EIEIO");
#endif
#if defined (EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
  TEST_COMPARE_STRING (strerrordesc_np (EWOULDBLOCK),
		       "Operation would block");
  TEST_COMPARE_STRING (strerrorname_np (EWOULDBLOCK), "EWOULDBLOCK");
#endif
#ifdef ENEEDAUTH
  TEST_COMPARE_STRING (strerrordesc_np (ENEEDAUTH), "Need authenticator");
  TEST_COMPARE_STRING (strerrorname_np (ENEEDAUTH), "ENEEDAUTH");
#endif
#ifdef ED
  TEST_COMPARE_STRING (strerrordesc_np (ED), "?");
  TEST_COMPARE_STRING (strerrorname_np (ED), "ED");
#endif
#ifdef EPROGUNAVAIL
  TEST_COMPARE_STRING (strerrordesc_np (EPROGUNAVAIL),
		       "RPC program not available");
  TEST_COMPARE_STRING (strerrorname_np (EPROGUNAVAIL), "EPROGUNAVAIL");
#endif

  return 0;
}

#include <support/test-driver.c>
