//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

﻿#pragma once
/*
Referenced from Linux Kernel Header File
*/

#define __WINE_UNIX_ERRNO_SUCCESS 0       /* No error */
#define __WINE_UNIX_ERRNO_EPERM 1         /* Operation not permitted */
#define __WINE_UNIX_ERRNO_ENOENT 2        /* No such file or directory */
#define __WINE_UNIX_ERRNO_ESRCH 3         /* No such process */
#define __WINE_UNIX_ERRNO_EINTR 4         /* Interrupted system call */
#define __WINE_UNIX_ERRNO_EIO 5           /* I/O error */
#define __WINE_UNIX_ERRNO_ENXIO 6         /* No such device or address */
#define __WINE_UNIX_ERRNO_E2BIG 7         /* Argument list too long */
#define __WINE_UNIX_ERRNO_ENOEXEC 8       /* Exec format error */
#define __WINE_UNIX_ERRNO_EBADF 9         /* Bad file number */
#define __WINE_UNIX_ERRNO_ECHILD 10       /* No child processes */
#define __WINE_UNIX_ERRNO_EAGAIN 11       /* Try again */
#define __WINE_UNIX_ERRNO_ENOMEM 12       /* Out of memory */
#define __WINE_UNIX_ERRNO_EACCES 13       /* Permission denied */
#define __WINE_UNIX_ERRNO_EFAULT 14       /* Bad address */
#define __WINE_UNIX_ERRNO_ENOTBLK 15      /* Block device required */
#define __WINE_UNIX_ERRNO_EBUSY 16        /* Device or resource busy */
#define __WINE_UNIX_ERRNO_EEXIST 17       /* File exists */
#define __WINE_UNIX_ERRNO_EXDEV 18        /* Cross-device link */
#define __WINE_UNIX_ERRNO_ENODEV 19       /* No such device */
#define __WINE_UNIX_ERRNO_ENOTDIR 20      /* Not a directory */
#define __WINE_UNIX_ERRNO_EISDIR 21       /* Is a directory */
#define __WINE_UNIX_ERRNO_EINVAL 22       /* Invalid argument */
#define __WINE_UNIX_ERRNO_ENFILE 23       /* File table overflow */
#define __WINE_UNIX_ERRNO_EMFILE 24       /* Too many open files */
#define __WINE_UNIX_ERRNO_ENOTTY 25       /* Not a typewriter */
#define __WINE_UNIX_ERRNO_ETXTBSY 26      /* Text file busy */
#define __WINE_UNIX_ERRNO_EFBIG 27        /* File too large */
#define __WINE_UNIX_ERRNO_ENOSPC 28       /* No space left on device */
#define __WINE_UNIX_ERRNO_ESPIPE 29       /* Illegal seek */
#define __WINE_UNIX_ERRNO_EROFS 30        /* Read-only file system */
#define __WINE_UNIX_ERRNO_EMLINK 31       /* Too many links */
#define __WINE_UNIX_ERRNO_EPIPE 32        /* Broken pipe */
#define __WINE_UNIX_ERRNO_EDOM 33         /* Math argument out of domain of func */
#define __WINE_UNIX_ERRNO_ERANGE 34       /* Math result not representable */
#define __WINE_UNIX_ERRNO_EDEADLK 35      /* Resource deadlock would occur */
#define __WINE_UNIX_ERRNO_ENAMETOOLONG 36 /* File name too long */
#define __WINE_UNIX_ERRNO_ENOLCK 37       /* No record locks available */

/*
 * This error code is special: arch syscall entry code will return
 * -ENOSYS if users try to call a syscall that doesn't exist.  To keep
 * failures of syscalls that really do exist distinguishable from
 * failures due to attempts to use a nonexistent syscall, syscall
 * implementations should refrain from returning -ENOSYS.
 */
#define __WINE_UNIX_ERRNO_ENOSYS 38 /* Invalid system call number */

#define __WINE_UNIX_ERRNO_ENOTEMPTY 39   /* Directory not empty */
#define __WINE_UNIX_ERRNO_ELOOP 40       /* Too many symbolic links encountered */
#define __WINE_UNIX_ERRNO_EWOULDBLOCK 41 /* Operation would block */
#define __WINE_UNIX_ERRNO_ENOMSG 42      /* No message of desired type */
#define __WINE_UNIX_ERRNO_EIDRM 43       /* Identifier removed */
#define __WINE_UNIX_ERRNO_ECHRNG 44      /* Channel number out of range */
#define __WINE_UNIX_ERRNO_EL2NSYNC 45    /* Level 2 not synchronized */
#define __WINE_UNIX_ERRNO_EL3HLT 46      /* Level 3 halted */
#define __WINE_UNIX_ERRNO_EL3RST 47      /* Level 3 reset */
#define __WINE_UNIX_ERRNO_ELNRNG 48      /* Link number out of range */
#define __WINE_UNIX_ERRNO_EUNATCH 49     /* Protocol driver not attached */
#define __WINE_UNIX_ERRNO_ENOCSI 50      /* No CSI structure available */
#define __WINE_UNIX_ERRNO_EL2HLT 51      /* Level 2 halted */
#define __WINE_UNIX_ERRNO_EBADE 52       /* Invalid exchange */
#define __WINE_UNIX_ERRNO_EBADR 53       /* Invalid request descriptor */
#define __WINE_UNIX_ERRNO_EXFULL 54      /* Exchange full */
#define __WINE_UNIX_ERRNO_ENOANO 55      /* No anode */
#define __WINE_UNIX_ERRNO_EBADRQC 56     /* Invalid request code */
#define __WINE_UNIX_ERRNO_EBADSLT 57     /* Invalid slot */

#define __WINE_UNIX_ERRNO_EDEADLOCK 58

#define __WINE_UNIX_ERRNO_EBFONT 59          /* Bad font file format */
#define __WINE_UNIX_ERRNO_ENOSTR 60          /* Device not a stream */
#define __WINE_UNIX_ERRNO_ENODATA 61         /* No data available */
#define __WINE_UNIX_ERRNO_ETIME 62           /* Timer expired */
#define __WINE_UNIX_ERRNO_ENOSR 63           /* Out of streams resources */
#define __WINE_UNIX_ERRNO_ENONET 64          /* Machine is not on the network */
#define __WINE_UNIX_ERRNO_ENOPKG 65          /* Package not installed */
#define __WINE_UNIX_ERRNO_EREMOTE 66         /* Object is remote */
#define __WINE_UNIX_ERRNO_ENOLINK 67         /* Link has been severed */
#define __WINE_UNIX_ERRNO_EADV 68            /* Advertise error */
#define __WINE_UNIX_ERRNO_ESRMNT 69          /* Srmount error */
#define __WINE_UNIX_ERRNO_ECOMM 70           /* Communication error on send */
#define __WINE_UNIX_ERRNO_EPROTO 71          /* Protocol error */
#define __WINE_UNIX_ERRNO_EMULTIHOP 72       /* Multihop attempted */
#define __WINE_UNIX_ERRNO_EDOTDOT 73         /* RFS specific error */
#define __WINE_UNIX_ERRNO_EBADMSG 74         /* Not a data message */
#define __WINE_UNIX_ERRNO_EOVERFLOW 75       /* Value too large for defined data type */
#define __WINE_UNIX_ERRNO_ENOTUNIQ 76        /* Name not unique on network */
#define __WINE_UNIX_ERRNO_EBADFD 77          /* File descriptor in bad state */
#define __WINE_UNIX_ERRNO_EREMCHG 78         /* Remote address changed */
#define __WINE_UNIX_ERRNO_ELIBACC 79         /* Can not access a needed shared library */
#define __WINE_UNIX_ERRNO_ELIBBAD 80         /* Accessing a corrupted shared library */
#define __WINE_UNIX_ERRNO_ELIBSCN 81         /* .lib section in a.out corrupted */
#define __WINE_UNIX_ERRNO_ELIBMAX 82         /* Attempting to link in too many shared libraries */
#define __WINE_UNIX_ERRNO_ELIBEXEC 83        /* Cannot exec a shared library directly */
#define __WINE_UNIX_ERRNO_EILSEQ 84          /* Illegal byte sequence */
#define __WINE_UNIX_ERRNO_ERESTART 85        /* Interrupted system call should be restarted */
#define __WINE_UNIX_ERRNO_ESTRPIPE 86        /* Streams pipe error */
#define __WINE_UNIX_ERRNO_EUSERS 87          /* Too many users */
#define __WINE_UNIX_ERRNO_ENOTSOCK 88        /* Socket operation on non-socket */
#define __WINE_UNIX_ERRNO_EDESTADDRREQ 89    /* Destination address required */
#define __WINE_UNIX_ERRNO_EMSGSIZE 90        /* Message too long */
#define __WINE_UNIX_ERRNO_EPROTOTYPE 91      /* Protocol wrong type for socket */
#define __WINE_UNIX_ERRNO_ENOPROTOOPT 92     /* Protocol not available */
#define __WINE_UNIX_ERRNO_EPROTONOSUPPORT 93 /* Protocol not supported */
#define __WINE_UNIX_ERRNO_ESOCKTNOSUPPORT 94 /* Socket type not supported */
#define __WINE_UNIX_ERRNO_EOPNOTSUPP 95      /* Operation not supported on transport endpoint */
#define __WINE_UNIX_ERRNO_EPFNOSUPPORT 96    /* Protocol family not supported */
#define __WINE_UNIX_ERRNO_EAFNOSUPPORT 97    /* Address family not supported by protocol */
#define __WINE_UNIX_ERRNO_EADDRINUSE 98      /* Address already in use */
#define __WINE_UNIX_ERRNO_EADDRNOTAVAIL 99   /* Cannot assign requested address */
#define __WINE_UNIX_ERRNO_ENETDOWN 100       /* Network is down */
#define __WINE_UNIX_ERRNO_ENETUNREACH 101    /* Network is unreachable */
#define __WINE_UNIX_ERRNO_ENETRESET 102      /* Network dropped connection because of reset */
#define __WINE_UNIX_ERRNO_ECONNABORTED 103   /* Software caused connection abort */
#define __WINE_UNIX_ERRNO_ECONNRESET 104     /* Connection reset by peer */
#define __WINE_UNIX_ERRNO_ENOBUFS 105        /* No buffer space available */
#define __WINE_UNIX_ERRNO_EISCONN 106        /* Transport endpoint is already connected */
#define __WINE_UNIX_ERRNO_ENOTCONN 107       /* Transport endpoint is not connected */
#define __WINE_UNIX_ERRNO_ESHUTDOWN 108      /* Cannot send after transport endpoint shutdown */
#define __WINE_UNIX_ERRNO_ETOOMANYREFS 109   /* Too many references: cannot splice */
#define __WINE_UNIX_ERRNO_ETIMEDOUT 110      /* Connection timed out */
#define __WINE_UNIX_ERRNO_ECONNREFUSED 111   /* Connection refused */
#define __WINE_UNIX_ERRNO_EHOSTDOWN 112      /* Host is down */
#define __WINE_UNIX_ERRNO_EHOSTUNREACH 113   /* No route to host */
#define __WINE_UNIX_ERRNO_EALREADY 114       /* Operation already in progress */
#define __WINE_UNIX_ERRNO_EINPROGRESS 115    /* Operation now in progress */
#define __WINE_UNIX_ERRNO_ESTALE 116         /* Stale file handle */
#define __WINE_UNIX_ERRNO_EUCLEAN 117        /* Structure needs cleaning */
#define __WINE_UNIX_ERRNO_ENOTNAM 118        /* Not a XENIX named type file */
#define __WINE_UNIX_ERRNO_ENAVAIL 119        /* No XENIX semaphores available */
#define __WINE_UNIX_ERRNO_EISNAM 120         /* Is a named type file */
#define __WINE_UNIX_ERRNO_EREMOTEIO 121      /* Remote I/O error */
#define __WINE_UNIX_ERRNO_EDQUOT 122         /* Quota exceeded */

#define __WINE_UNIX_ERRNO_ENOMEDIUM 123    /* No medium found */
#define __WINE_UNIX_ERRNO_EMEDIUMTYPE 124  /* Wrong medium type */
#define __WINE_UNIX_ERRNO_ECANCELED 125    /* Operation Canceled */
#define __WINE_UNIX_ERRNO_ENOKEY 126       /* Required key not available */
#define __WINE_UNIX_ERRNO_EKEYEXPIRED 127  /* Key has expired */
#define __WINE_UNIX_ERRNO_EKEYREVOKED 128  /* Key has been revoked */
#define __WINE_UNIX_ERRNO_EKEYREJECTED 129 /* Key was rejected by service */

/* for robust mutexes */
#define __WINE_UNIX_ERRNO_EOWNERDEAD 130      /* Owner died */
#define __WINE_UNIX_ERRNO_ENOTRECOVERABLE 131 /* State not recoverable */

#define __WINE_UNIX_ERRNO_ERFKILL 132 /* Operation not possible due to RF-kill */

#define __WINE_UNIX_ERRNO_EHWPOISON 133 /* Memory page has hardware error */
