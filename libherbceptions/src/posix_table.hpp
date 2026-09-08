// clang-format off
#define POSIX_ERRC_MAX_SIZE 61

	case 0:
		return __tsc(u8"Success");
#if defined(EPERM) && (!defined(EACCES) || (EPERM != EACCES))
	case EPERM:
		return __tsc(u8"Not owner");
#endif
#ifdef ENOENT
	case ENOENT:
		return __tsc(u8"No such file or directory");
#endif
#ifdef ESRCH
	case ESRCH:
		return __tsc(u8"No such process");
#endif
#ifdef EINTR
	case EINTR:
		return __tsc(u8"Interrupted system call");
#endif
#ifdef EIO
	case EIO:
		return __tsc(u8"I/O error");
#endif
/* go32 defines ENXIO as ENODEV */
#if defined(ENXIO) && (!defined(ENODEV) || (ENXIO != ENODEV))
	case ENXIO:
		return __tsc(u8"No such device or address");
#endif
#ifdef E2BIG
	case E2BIG:
		return __tsc(u8"Arg list too long");
#endif
#ifdef ENOEXEC
	case ENOEXEC:
		return __tsc(u8"Exec format error");
#endif
#ifdef EALREADY
	case EALREADY:
		return __tsc(u8"Socket already connected");
#endif
#ifdef EBADF
	case EBADF:
		return __tsc(u8"Bad file number");
#endif
#ifdef ECHILD
	case ECHILD:
		return __tsc(u8"No children");
#endif
#ifdef EDESTADDRREQ
	case EDESTADDRREQ:
		return __tsc(u8"Destination address required");
#endif
#ifdef EAGAIN
	case EAGAIN:
		return __tsc(u8"No more processes");
#endif
#ifdef ENOMEM
	case ENOMEM:
		return __tsc(u8"Not enough space");
#endif
#ifdef EACCES
	case EACCES:
		return __tsc(u8"Permission denied");
#endif
#ifdef EFAULT
	case EFAULT:
		return __tsc(u8"Bad address");
#endif
#ifdef ENOTBLK
	case ENOTBLK:
		return __tsc(u8"Block device required");
#endif
#ifdef EBUSY
	case EBUSY:
		return __tsc(u8"Device or resource busy");
#endif
#ifdef EEXIST
	case EEXIST:
		return __tsc(u8"File exists");
#endif
#ifdef EXDEV
	case EXDEV:
		return __tsc(u8"Cross-device link");
#endif
#ifdef ENODEV
	case ENODEV:
		return __tsc(u8"No such device");
#endif
#ifdef ENOTDIR
	case ENOTDIR:
		return __tsc(u8"Not a directory");
#endif
#ifdef EHOSTDOWN
	case EHOSTDOWN:
		return __tsc(u8"Host is down");
#endif
#ifdef EINPROGRESS
	case EINPROGRESS:
		return __tsc(u8"Connection already in progress");
#endif
#ifdef EISDIR
	case EISDIR:
		return __tsc(u8"Is a directory");
#endif
#ifdef EINVAL
	case EINVAL:
		return __tsc(u8"Invalid argument");
#endif
#ifdef ENETDOWN
	case ENETDOWN:
		return __tsc(u8"Network interface is not configured");
#endif
#ifdef ENETRESET
	case ENETRESET:
		return __tsc(u8"Connection aborted by network");
#endif
#ifdef ENFILE
	case ENFILE:
		return __tsc(u8"Too many open files in system");
#endif
#ifdef EMFILE
	case EMFILE:
		return __tsc(u8"File descriptor value too large");
#endif
#ifdef ENOTTY
	case ENOTTY:
		return __tsc(u8"Not a character device");
#endif
#ifdef ETXTBSY
	case ETXTBSY:
		return __tsc(u8"Text file busy");
#endif
#ifdef EFBIG
	case EFBIG:
		return __tsc(u8"File too large");
#endif
#ifdef EHOSTUNREACH
	case EHOSTUNREACH:
		return __tsc(u8"Host is unreachable");
#endif
#ifdef ENOSPC
	case ENOSPC:
		return __tsc(u8"No space left on device");
#endif
#ifdef ENOTSUP
	case ENOTSUP:
		return __tsc(u8"Not supported");
#endif
#ifdef ESPIPE
	case ESPIPE:
		return __tsc(u8"Illegal seek");
#endif
#ifdef EROFS
	case EROFS:
		return __tsc(u8"Read-only file system");
#endif
#ifdef EMLINK
	case EMLINK:
		return __tsc(u8"Too many links");
#endif
#ifdef EPIPE
	case EPIPE:
		return __tsc(u8"Broken pipe");
#endif
#ifdef EDOM
	case EDOM:
		return __tsc(u8"Mathematics argument out of domain of function");
#endif
#ifdef ERANGE
	case ERANGE:
		return __tsc(u8"Result too large");
#endif
#ifdef ENOMSG
	case ENOMSG:
		return __tsc(u8"No message of desired type");
#endif
#ifdef EIDRM
	case EIDRM:
		return __tsc(u8"Identifier removed");
#endif
#ifdef EILSEQ
	case EILSEQ:
		return __tsc(u8"Illegal byte sequence");
#endif
#ifdef EDEADLK
	case EDEADLK:
		return __tsc(u8"Deadlock");
#endif
#ifdef ENETUNREACH
	case ENETUNREACH:
		return __tsc(u8"Network is unreachable");
#endif
#ifdef ENOLCK
	case ENOLCK:
		return __tsc(u8"No lock");
#endif
#ifdef ENOSTR
	case ENOSTR:
		return __tsc(u8"Not a stream");
#endif
#ifdef ETIME
	case ETIME:
		return __tsc(u8"Stream ioctl timeout");
#endif
#ifdef ENOSR
	case ENOSR:
		return __tsc(u8"No stream resources");
#endif
#ifdef ENONET
	case ENONET:
		return __tsc(u8"Machine is not on the network");
#endif
#ifdef ENOPKG
	case ENOPKG:
		return __tsc(u8"No package");
#endif
#ifdef EREMOTE
	case EREMOTE:
		return __tsc(u8"Resource is remote");
#endif
#ifdef ENOLINK
	case ENOLINK:
		return __tsc(u8"Virtual circuit is gone");
#endif
#ifdef EADV
	case EADV:
		return __tsc(u8"Advertise error");
#endif
#ifdef ESRMNT
	case ESRMNT:
		return __tsc(u8"Srmount error");
#endif
#ifdef ECOMM
	case ECOMM:
		return __tsc(u8"Communication error");
#endif
#ifdef EPROTO
	case EPROTO:
		return __tsc(u8"Protocol error");
#endif
#ifdef EPROTONOSUPPORT
	case EPROTONOSUPPORT:
		return __tsc(u8"Unknown protocol");
#endif
#ifdef EMULTIHOP
	case EMULTIHOP:
		return __tsc(u8"Multihop attempted");
#endif
#ifdef EBADMSG
	case EBADMSG:
		return __tsc(u8"Bad message");
#endif
#ifdef ELIBACC
	case ELIBACC:
		return __tsc(u8"Cannot access a needed shared library");
#endif
#ifdef ELIBBAD
	case ELIBBAD:
		return __tsc(u8"Accessing a corrupted shared library");
#endif
#ifdef ELIBSCN
	case ELIBSCN:
		return __tsc(u8".lib section in a.out corrupted");
#endif
#ifdef ELIBMAX
	case ELIBMAX:
		return __tsc(u8"Attempting to link in more shared libraries than system limit");
#endif
#ifdef ELIBEXEC
	case ELIBEXEC:
		return __tsc(u8"Cannot exec a shared library directly");
#endif
#ifdef ENOSYS
	case ENOSYS:
		return __tsc(u8"Function not implemented");
#endif
#ifdef ENMFILE
	case ENMFILE:
		return __tsc(u8"No more files");
#endif
#ifdef ENOTEMPTY
	case ENOTEMPTY:
		return __tsc(u8"Directory not empty");
#endif
#ifdef ENAMETOOLONG
	case ENAMETOOLONG:
		return __tsc(u8"File or path name too long");
#endif
#ifdef ELOOP
	case ELOOP:
		return __tsc(u8"Too many symbolic links");
#endif
#ifdef ENOBUFS
	case ENOBUFS:
		return __tsc(u8"No buffer space available");
#endif
#ifdef ENODATA
	case ENODATA:
		return __tsc(u8"No data");
#endif
#ifdef EAFNOSUPPORT
	case EAFNOSUPPORT:
		return __tsc(u8"Address family not supported by protocol family");
#endif
#ifdef EPROTOTYPE
	case EPROTOTYPE:
		return __tsc(u8"Protocol wrong type for socket");
#endif
#ifdef ENOTSOCK
	case ENOTSOCK:
		return __tsc(u8"Socket operation on non-socket");
#endif
#ifdef ENOPROTOOPT
	case ENOPROTOOPT:
		return __tsc(u8"Protocol not available");
#endif
#ifdef ESHUTDOWN
	case ESHUTDOWN:
		return __tsc(u8"Can't send after socket shutdown");
#endif
#ifdef ECONNREFUSED
	case ECONNREFUSED:
		return __tsc(u8"Connection refused");
#endif
#ifdef ECONNRESET
	case ECONNRESET:
		return __tsc(u8"Connection reset by peer");
#endif
#ifdef EADDRINUSE
	case EADDRINUSE:
		return __tsc(u8"Address already in use");
#endif
#ifdef EADDRNOTAVAIL
	case EADDRNOTAVAIL:
		return __tsc(u8"Address not available");
#endif
#ifdef ECONNABORTED
	case ECONNABORTED:
		return __tsc(u8"Software caused connection abort");
#endif
#if (defined(EWOULDBLOCK) && (!defined(EAGAIN) || (EWOULDBLOCK != EAGAIN)))
	case EWOULDBLOCK:
		return __tsc(u8"Operation would block");
#endif
#ifdef ENOTCONN
	case ENOTCONN:
		return __tsc(u8"Socket is not connected");
#endif
#ifdef ESOCKTNOSUPPORT
	case ESOCKTNOSUPPORT:
		return __tsc(u8"Socket type not supported");
#endif
#ifdef EISCONN
	case EISCONN:
		return __tsc(u8"Socket is already connected");
#endif
#ifdef ECANCELED
	case ECANCELED:
		return __tsc(u8"Operation canceled");
#endif
#ifdef ENOTRECOVERABLE
	case ENOTRECOVERABLE:
		return __tsc(u8"State not recoverable");
#endif
#ifdef EOWNERDEAD
	case EOWNERDEAD:
		return __tsc(u8"Previous owner died");
#endif
#ifdef ESTRPIPE
	case ESTRPIPE:
		return __tsc(u8"Streams pipe error");
#endif
#if defined(EOPNOTSUPP) && (!defined(ENOTSUP) || (ENOTSUP != EOPNOTSUPP))
	case EOPNOTSUPP:
		return __tsc(u8"Operation not supported on socket");
#endif
#ifdef EOVERFLOW
	case EOVERFLOW:
		return __tsc(u8"Value too large for defined data type");
#endif
#ifdef EMSGSIZE
	case EMSGSIZE:
		return __tsc(u8"Message too long");
#endif
#ifdef ETIMEDOUT
	case ETIMEDOUT:
		return __tsc(u8"Connection timed out");
#endif
	default:
		return __tsc(u8"Unknown");
// clang-format on
