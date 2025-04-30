/* System-specific socket constants and types.  Hurd version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef __BITS_SOCKET_H
#define __BITS_SOCKET_H	1

#ifndef _SYS_SOCKET_H
# error "Never include <bits/socket.h> directly; use <sys/socket.h> instead."
#endif

#define	__need_size_t
#include <stddef.h>

#include <bits/wordsize.h>
#include <sys/types.h>

/* Type for length arguments in socket calls.  */
#ifndef __socklen_t_defined
typedef __socklen_t socklen_t;
# define __socklen_t_defined
#endif


/* Types of sockets.  */
enum __socket_type
{
  SOCK_STREAM = 1,		/* Sequenced, reliable, connection-based
				   byte streams.  */
#define SOCK_STREAM SOCK_STREAM
  SOCK_DGRAM = 2,		/* Connectionless, unreliable datagrams
				   of fixed maximum length.  */
#define SOCK_DGRAM SOCK_DGRAM
  SOCK_RAW = 3,			/* Raw protocol interface.  */
#define SOCK_RAW SOCK_RAW
  SOCK_RDM = 4,			/* Reliably-delivered messages.  */
#define SOCK_RDM SOCK_RDM
  SOCK_SEQPACKET = 5,		/* Sequenced, reliable, connection-based,
				   datagrams of fixed maximum length.  */
#define SOCK_SEQPACKET SOCK_SEQPACKET

#define SOCK_MAX (SOCK_SEQPACKET + 1)
  /* Mask which covers at least up to SOCK_MASK-1.  The
     remaining bits are used as flags. */
#define SOCK_TYPE_MASK 0xf

  /* Flags to be ORed into the type parameter of socket and socketpair and
     used for the flags parameter of accept4.  */

  SOCK_CLOEXEC = 0x00400000,	/* Atomically set close-on-exec flag for the
				   new descriptor(s).  */
#define SOCK_CLOEXEC SOCK_CLOEXEC

  /* Changed from the O_NONBLOCK value (0x8, which is unusable for us as it is
     conflicting with the original SOCK_* flags' values) to the Linux value
     (04000).  TODO: is there a ``better'' value?  */
  SOCK_NONBLOCK = 0x0800	/* Atomically mark descriptor(s) as
				   non-blocking.  */
#define SOCK_NONBLOCK SOCK_NONBLOCK
};

/* Protocol families.  */
#define	PF_UNSPEC	0	/* Unspecified.  */
#define	PF_LOCAL	1	/* Local to host (pipes and file-domain).  */
#define	PF_UNIX		PF_LOCAL /* Old BSD name for PF_LOCAL.  */
#define	PF_FILE		PF_LOCAL /* POSIX name for PF_LOCAL.  */
#define	PF_INET		2	/* IP protocol family.  */
#define	PF_IMPLINK	3	/* ARPAnet IMP protocol.  */
#define	PF_PUP		4	/* PUP protocols.  */
#define	PF_CHAOS	5	/* MIT Chaos protocols.  */
#define	PF_NS		6	/* Xerox NS protocols.  */
#define	PF_ISO		7	/* ISO protocols.  */
#define	PF_OSI		PF_ISO
#define	PF_ECMA		8	/* ECMA protocols.  */
#define	PF_DATAKIT	9	/* AT&T Datakit protocols.  */
#define	PF_CCITT	10	/* CCITT protocols (X.25 et al).  */
#define	PF_SNA		11	/* IBM SNA protocol.  */
#define	PF_DECnet	12	/* DECnet protocols.  */
#define	PF_DLI		13	/* Direct data link interface.  */
#define	PF_LAT		14	/* DEC Local Area Transport protocol.  */
#define	PF_HYLINK	15	/* NSC Hyperchannel protocol.  */
#define	PF_APPLETALK	16	/* Don't use this.  */
#define	PF_ROUTE	17	/* Internal Routing Protocol.  */
#define	PF_LINK		18	/* Link layer interface.  */
#define	PF_XTP		19	/* eXpress Transfer Protocol (no AF).  */
#define	PF_COIP		20	/* Connection-oriented IP, aka ST II.  */
#define	PF_CNT		21	/* Computer Network Technology.  */
#define PF_RTIP		22	/* Help Identify RTIP packets.  **/
#define	PF_IPX		23	/* Novell Internet Protocol.  */
#define	PF_SIP		24	/* Simple Internet Protocol.  */
#define PF_PIP		25	/* Help Identify PIP packets.  */
#define PF_INET6	26	/* IP version 6.  */
#define	PF_MAX		27

/* Address families.  */
#define	AF_UNSPEC	PF_UNSPEC
#define	AF_LOCAL	PF_LOCAL
#define	AF_UNIX		PF_UNIX
#define	AF_FILE		PF_FILE
#define	AF_INET		PF_INET
#define	AF_IMPLINK	PF_IMPLINK
#define	AF_PUP		PF_PUP
#define	AF_CHAOS	PF_CHAOS
#define	AF_NS		PF_NS
#define	AF_ISO		PF_ISO
#define	AF_OSI		PF_OSI
#define	AF_ECMA		PF_ECMA
#define	AF_DATAKIT	PF_DATAKIT
#define	AF_CCITT	PF_CCITT
#define	AF_SNA		PF_SNA
#define	AF_DECnet	PF_DECnet
#define	AF_DLI		PF_DLI
#define	AF_LAT		PF_LAT
#define	AF_HYLINK	PF_HYLINK
#define	AF_APPLETALK	PF_APPLETALK
#define	AF_ROUTE	PF_ROUTE
#define	AF_LINK		PF_LINK
#ifdef __USE_MISC
# define	pseudo_AF_XTP	PF_XTP
#endif
#define	AF_COIP		PF_COIP
#define	AF_CNT		PF_CNT
#ifdef __USE_MISC
# define pseudo_AF_RTIP	PF_RTIP
#endif
#define	AF_IPX		PF_IPX
#define	AF_SIP		PF_SIP
#ifdef __USE_MISC
# define pseudo_AF_PIP	PF_PIP
#endif
#define AF_INET6	PF_INET6
#define	AF_MAX		PF_MAX

/* Maximum queue length specifiable by listen.  */
#define SOMAXCONN	128	/* 5 on the origional 4.4 BSD.  */

/* Get the definition of the macro to define the common sockaddr members.  */
#include <bits/sockaddr.h>

/* Structure describing a generic socket address.  */
struct sockaddr
  {
    __SOCKADDR_COMMON (sa_);	/* Common data: address family and length.  */
    char sa_data[14];		/* Address data.  */
  };


/* Structure large enough to hold any socket address (with the historical
   exception of AF_UNIX).  */
#if __WORDSIZE == 64
# define __ss_aligntype	__uint64_t
#else
# define __ss_aligntype	__uint32_t
#endif
#define _SS_PADSIZE \
  (_SS_SIZE - __SOCKADDR_COMMON_SIZE - sizeof (__ss_aligntype))

struct sockaddr_storage
  {
    __SOCKADDR_COMMON (ss_);	/* Address family, etc.  */
    char __ss_padding[_SS_PADSIZE];
    __ss_aligntype __ss_align;	/* Force desired alignment.  */
  };


/* Bits in the FLAGS argument to `send', `recv', et al.  */
enum
  {
    MSG_OOB		= 0x01,	/* Process out-of-band data.  */
#define MSG_OOB MSG_OOB
    MSG_PEEK		= 0x02,	/* Peek at incoming messages.  */
#define MSG_PEEK MSG_PEEK
    MSG_DONTROUTE	= 0x04,	/* Don't use local routing.  */
#define MSG_DONTROUTE MSG_DONTROUTE
    MSG_EOR		= 0x08,	/* Data completes record.  */
#define MSG_EOR MSG_EOR
    MSG_TRUNC		= 0x10,	/* Data discarded before delivery.  */
#define MSG_TRUNC MSG_TRUNC
    MSG_CTRUNC		= 0x20,	/* Control data lost before delivery.  */
#define MSG_CTRUNC MSG_CTRUNC
    MSG_WAITALL		= 0x40,	/* Wait for full request or error.  */
#define MSG_WAITALL MSG_WAITALL
    MSG_DONTWAIT	= 0x80,	/* This message should be nonblocking.  */
#define MSG_DONTWAIT MSG_DONTWAIT
    MSG_NOSIGNAL	= 0x0400	/* Do not generate SIGPIPE on EPIPE.  */
#define MSG_NOSIGNAL MSG_NOSIGNAL
  };


/* Structure describing messages sent by
   `sendmsg' and received by `recvmsg'.  */
struct msghdr
  {
    void *msg_name;		/* Address to send to/receive from.  */
    socklen_t msg_namelen;	/* Length of address data.  */

    struct iovec *msg_iov;	/* Vector of data to send/receive into.  */
    int msg_iovlen;		/* Number of elements in the vector.  */

    void *msg_control;		/* Ancillary data (eg BSD filedesc passing). */
    socklen_t msg_controllen;	/* Ancillary data buffer length.  */

    int msg_flags;		/* Flags in received message.  */
  };

/* Structure used for storage of ancillary data object information.  */
struct cmsghdr
  {
    socklen_t cmsg_len;		/* Length of data in cmsg_data plus length
				   of cmsghdr structure.  */
    int cmsg_level;		/* Originating protocol.  */
    int cmsg_type;		/* Protocol specific type.  */
#if __glibc_c99_flexarr_available
    __extension__ unsigned char __cmsg_data __flexarr; /* Ancillary data.  */
#endif
  };

/* Ancillary data object manipulation macros.  */
#if __glibc_c99_flexarr_available
# define CMSG_DATA(cmsg) ((cmsg)->__cmsg_data)
#else
# define CMSG_DATA(cmsg) ((unsigned char *) ((struct cmsghdr *) (cmsg) + 1))
#endif

#define CMSG_NXTHDR(mhdr, cmsg) __cmsg_nxthdr (mhdr, cmsg)

#define CMSG_FIRSTHDR(mhdr) \
  ((size_t) (mhdr)->msg_controllen >= sizeof (struct cmsghdr)		      \
   ? (struct cmsghdr *) (mhdr)->msg_control : (struct cmsghdr *) 0)

#define CMSG_ALIGN(len) (((len) + sizeof (size_t) - 1) \
			   & (size_t) ~(sizeof (size_t) - 1))
#define CMSG_SPACE(len) (CMSG_ALIGN (len) \
			 + CMSG_ALIGN (sizeof (struct cmsghdr)))
#define CMSG_LEN(len)   (CMSG_ALIGN (sizeof (struct cmsghdr)) + (len))

extern struct cmsghdr *__cmsg_nxthdr (struct msghdr *__mhdr,
				      struct cmsghdr *__cmsg) __THROW;
#ifdef __USE_EXTERN_INLINES
# ifndef _EXTERN_INLINE
#  define _EXTERN_INLINE __extern_inline
# endif
_EXTERN_INLINE struct cmsghdr *
__NTH (__cmsg_nxthdr (struct msghdr *__mhdr, struct cmsghdr *__cmsg))
{
  if ((size_t) __cmsg->cmsg_len < sizeof (struct cmsghdr))
    /* The kernel header does this so there may be a reason.  */
    return (struct cmsghdr *) 0;

  __cmsg = (struct cmsghdr *) ((unsigned char *) __cmsg
			       + CMSG_ALIGN (__cmsg->cmsg_len));
  if ((unsigned char *) (__cmsg + 1) > ((unsigned char *) __mhdr->msg_control
					+ __mhdr->msg_controllen)
      || ((unsigned char *) __cmsg + CMSG_ALIGN (__cmsg->cmsg_len)
	  > ((unsigned char *) __mhdr->msg_control + __mhdr->msg_controllen)))
    /* No more entries.  */
    return (struct cmsghdr *) 0;
  return __cmsg;
}
#endif	/* Use `extern inline'.  */

/* Socket level message types.  */
enum
  {
    SCM_RIGHTS = 0x01,		/* Access rights (array of int).  */
#define SCM_RIGHTS SCM_RIGHTS
    SCM_TIMESTAMP = 0x02,	/* Timestamp (struct timeval).  */
#define SCM_TIMESTAMP SCM_TIMESTAMP
    SCM_CREDS = 0x03		/* Process creds (struct cmsgcred).  */
#define SCM_CREDS SCM_CREDS
  };

#ifdef __USE_MISC
/* Unfortunately, BSD practice dictates this structure be of fixed size.
   If there are more than CMGROUP_MAX groups, the list is truncated.
   (On GNU systems, the `cmcred_euid' field is just the first in the
   list of effective UIDs.)  */
#define CMGROUP_MAX	16

/* Structure delivered by SCM_CREDS.  This describes the identity of the
   sender of the data simultaneously received on the socket.  By BSD
   convention, this is included only when a sender on a AF_LOCAL socket
   sends cmsg data of this type and size; the sender's structure is
   ignored, and the system fills in the various IDs of the sender process.  */
struct cmsgcred
  {
    __pid_t cmcred_pid;
    __uid_t cmcred_uid;
    __uid_t cmcred_euid;
    __gid_t cmcred_gid;
    int cmcred_ngroups;
    __gid_t cmcred_groups[CMGROUP_MAX];
  };
#endif

/* Protocol number used to manipulate socket-level options
   with `getsockopt' and `setsockopt'.  */
#define	SOL_SOCKET	0xffff

/* Socket-level options for `getsockopt' and `setsockopt'.  */
enum
  {
    SO_DEBUG = 0x0001,		/* Record debugging information.  */
#define SO_DEBUG SO_DEBUG
    SO_ACCEPTCONN = 0x0002,	/* Accept connections on socket.  */
#define SO_ACCEPTCONN SO_ACCEPTCONN
    SO_REUSEADDR = 0x0004,	/* Allow reuse of local addresses.  */
#define SO_REUSEADDR SO_REUSEADDR
    SO_KEEPALIVE = 0x0008,	/* Keep connections alive and send
				   SIGPIPE when they die.  */
#define SO_KEEPALIVE SO_KEEPALIVE
    SO_DONTROUTE = 0x0010,	/* Don't do local routing.  */
#define SO_DONTROUTE SO_DONTROUTE
    SO_BROADCAST = 0x0020,	/* Allow transmission of
				   broadcast messages.  */
#define SO_BROADCAST SO_BROADCAST
    SO_USELOOPBACK = 0x0040,	/* Use the software loopback to avoid
				   hardware use when possible.  */
#define SO_USELOOPBACK SO_USELOOPBACK
    SO_LINGER = 0x0080,		/* Block on close of a reliable
				   socket to transmit pending data.  */
#define SO_LINGER SO_LINGER
    SO_OOBINLINE = 0x0100,	/* Receive out-of-band data in-band.  */
#define SO_OOBINLINE SO_OOBINLINE
    SO_REUSEPORT = 0x0200,	/* Allow local address and port reuse.  */
#define SO_REUSEPORT SO_REUSEPORT
    SO_SNDBUF = 0x1001,		/* Send buffer size.  */
#define SO_SNDBUF SO_SNDBUF
    SO_RCVBUF = 0x1002,		/* Receive buffer.  */
#define SO_RCVBUF SO_RCVBUF
    SO_SNDLOWAT = 0x1003,	/* Send low-water mark.  */
#define SO_SNDLOWAT SO_SNDLOWAT
    SO_RCVLOWAT = 0x1004,	/* Receive low-water mark.  */
#define SO_RCVLOWAT SO_RCVLOWAT
    SO_SNDTIMEO = 0x1005,	/* Send timeout.  */
#define SO_SNDTIMEO SO_SNDTIMEO
    SO_RCVTIMEO = 0x1006,	/* Receive timeout.  */
#define SO_RCVTIMEO SO_RCVTIMEO
    SO_ERROR = 0x1007,		/* Get and clear error status.  */
#define SO_ERROR SO_ERROR
    SO_STYLE = 0x1008,		/* Get socket connection style.  */
#define SO_STYLE SO_STYLE
    SO_TYPE = SO_STYLE		/* Compatible name for SO_STYLE.  */
#define SO_TYPE SO_TYPE
  };

/* Structure used to manipulate the SO_LINGER option.  */
struct linger
  {
    int l_onoff;		/* Nonzero to linger on close.  */
    int l_linger;		/* Time to linger.  */
  };

#endif	/* bits/socket.h */
