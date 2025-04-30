_S(0, N_("Success"))
#ifdef EPERM
/*
TRANS Only the owner of the file (or other resource)
TRANS or processes with special privileges can perform the operation. */
_S(EPERM, N_("Operation not permitted"))
#endif
#ifdef ENOENT
/*
TRANS This is a ``file doesn't exist'' error
TRANS for ordinary files that are referenced in contexts where they are
TRANS expected to already exist. */
_S(ENOENT, N_("No such file or directory"))
#endif
#ifdef ESRCH
/*
TRANS No process matches the specified process ID. */
_S(ESRCH, N_("No such process"))
#endif
#ifdef EINTR
/*
TRANS An asynchronous signal occurred and prevented
TRANS completion of the call.  When this happens, you should try the call
TRANS again.
TRANS
TRANS You can choose to have functions resume after a signal that is handled,
TRANS rather than failing with @code{EINTR}; see @ref{Interrupted
TRANS Primitives}. */
_S(EINTR, N_("Interrupted system call"))
#endif
#ifdef EIO
/*
TRANS Usually used for physical read or write errors. */
_S(EIO, N_("Input/output error"))
#endif
#ifdef ENXIO
/*
TRANS The system tried to use the device
TRANS represented by a file you specified, and it couldn't find the device.
TRANS This can mean that the device file was installed incorrectly, or that
TRANS the physical device is missing or not correctly attached to the
TRANS computer. */
_S(ENXIO, N_("No such device or address"))
#endif
#ifdef E2BIG
/*
TRANS Used when the arguments passed to a new program
TRANS being executed with one of the @code{exec} functions (@pxref{Executing a
TRANS File}) occupy too much memory space.  This condition never arises on
TRANS @gnuhurdsystems{}. */
_S(E2BIG, N_("Argument list too long"))
#endif
#ifdef ENOEXEC
/*
TRANS Invalid executable file format.  This condition is detected by the
TRANS @code{exec} functions; see @ref{Executing a File}. */
_S(ENOEXEC, N_("Exec format error"))
#endif
#ifdef EBADF
/*
TRANS For example, I/O on a descriptor that has been
TRANS closed or reading from a descriptor open only for writing (or vice
TRANS versa). */
_S(EBADF, N_("Bad file descriptor"))
#endif
#ifdef ECHILD
/*
TRANS This error happens on operations that are
TRANS supposed to manipulate child processes, when there aren't any processes
TRANS to manipulate. */
_S(ECHILD, N_("No child processes"))
#endif
#ifdef EDEADLK
/*
TRANS Allocating a system resource would have resulted in a
TRANS deadlock situation.  The system does not guarantee that it will notice
TRANS all such situations.  This error means you got lucky and the system
TRANS noticed; it might just hang.  @xref{File Locks}, for an example. */
_S(EDEADLK, N_("Resource deadlock avoided"))
#endif
#ifdef ENOMEM
/*
TRANS The system cannot allocate more virtual memory
TRANS because its capacity is full. */
_S(ENOMEM, N_("Cannot allocate memory"))
#endif
#ifdef EACCES
/*
TRANS The file permissions do not allow the attempted operation. */
_S(EACCES, N_("Permission denied"))
#endif
#ifdef EFAULT
/*
TRANS An invalid pointer was detected.
TRANS On @gnuhurdsystems{}, this error never happens; you get a signal instead. */
_S(EFAULT, N_("Bad address"))
#endif
#ifdef ENOTBLK
/*
TRANS A file that isn't a block special file was given in a situation that
TRANS requires one.  For example, trying to mount an ordinary file as a file
TRANS system in Unix gives this error. */
_S(ENOTBLK, N_("Block device required"))
#endif
#ifdef EBUSY
/*
TRANS A system resource that can't be shared is already in use.
TRANS For example, if you try to delete a file that is the root of a currently
TRANS mounted filesystem, you get this error. */
_S(EBUSY, N_("Device or resource busy"))
#endif
#ifdef EEXIST
/*
TRANS An existing file was specified in a context where it only
TRANS makes sense to specify a new file. */
_S(EEXIST, N_("File exists"))
#endif
#ifdef EXDEV
/*
TRANS An attempt to make an improper link across file systems was detected.
TRANS This happens not only when you use @code{link} (@pxref{Hard Links}) but
TRANS also when you rename a file with @code{rename} (@pxref{Renaming Files}). */
_S(EXDEV, N_("Invalid cross-device link"))
#endif
#ifdef ENODEV
/*
TRANS The wrong type of device was given to a function that expects a
TRANS particular sort of device. */
_S(ENODEV, N_("No such device"))
#endif
#ifdef ENOTDIR
/*
TRANS A file that isn't a directory was specified when a directory is required. */
_S(ENOTDIR, N_("Not a directory"))
#endif
#ifdef EISDIR
/*
TRANS You cannot open a directory for writing,
TRANS or create or remove hard links to it. */
_S(EISDIR, N_("Is a directory"))
#endif
#ifdef EINVAL
/*
TRANS This is used to indicate various kinds of problems
TRANS with passing the wrong argument to a library function. */
_S(EINVAL, N_("Invalid argument"))
#endif
#ifdef EMFILE
/*
TRANS The current process has too many files open and can't open any more.
TRANS Duplicate descriptors do count toward this limit.
TRANS
TRANS In BSD and GNU, the number of open files is controlled by a resource
TRANS limit that can usually be increased.  If you get this error, you might
TRANS want to increase the @code{RLIMIT_NOFILE} limit or make it unlimited;
TRANS @pxref{Limits on Resources}. */
_S(EMFILE, N_("Too many open files"))
#endif
#ifdef ENFILE
/*
TRANS There are too many distinct file openings in the entire system.  Note
TRANS that any number of linked channels count as just one file opening; see
TRANS @ref{Linked Channels}.  This error never occurs on @gnuhurdsystems{}. */
_S(ENFILE, N_("Too many open files in system"))
#endif
#ifdef ENOTTY
/*
TRANS Inappropriate I/O control operation, such as trying to set terminal
TRANS modes on an ordinary file. */
_S(ENOTTY, N_("Inappropriate ioctl for device"))
#endif
#ifdef ETXTBSY
/*
TRANS An attempt to execute a file that is currently open for writing, or
TRANS write to a file that is currently being executed.  Often using a
TRANS debugger to run a program is considered having it open for writing and
TRANS will cause this error.  (The name stands for ``text file busy''.)  This
TRANS is not an error on @gnuhurdsystems{}; the text is copied as necessary. */
_S(ETXTBSY, N_("Text file busy"))
#endif
#ifdef EFBIG
/*
TRANS The size of a file would be larger than allowed by the system. */
_S(EFBIG, N_("File too large"))
#endif
#ifdef ENOSPC
/*
TRANS Write operation on a file failed because the
TRANS disk is full. */
_S(ENOSPC, N_("No space left on device"))
#endif
#ifdef ESPIPE
/*
TRANS Invalid seek operation (such as on a pipe). */
_S(ESPIPE, N_("Illegal seek"))
#endif
#ifdef EROFS
/*
TRANS An attempt was made to modify something on a read-only file system. */
_S(EROFS, N_("Read-only file system"))
#endif
#ifdef EMLINK
/*
TRANS The link count of a single file would become too large.
TRANS @code{rename} can cause this error if the file being renamed already has
TRANS as many links as it can take (@pxref{Renaming Files}). */
_S(EMLINK, N_("Too many links"))
#endif
#ifdef EPIPE
/*
TRANS There is no process reading from the other end of a pipe.
TRANS Every library function that returns this error code also generates a
TRANS @code{SIGPIPE} signal; this signal terminates the program if not handled
TRANS or blocked.  Thus, your program will never actually see @code{EPIPE}
TRANS unless it has handled or blocked @code{SIGPIPE}. */
_S(EPIPE, N_("Broken pipe"))
#endif
#ifdef EDOM
/*
TRANS Used by mathematical functions when an argument value does
TRANS not fall into the domain over which the function is defined. */
_S(EDOM, N_("Numerical argument out of domain"))
#endif
#ifdef ERANGE
/*
TRANS Used by mathematical functions when the result value is
TRANS not representable because of overflow or underflow. */
_S(ERANGE, N_("Numerical result out of range"))
#endif
#ifdef EAGAIN
/*
TRANS The call might work if you try again
TRANS later.  The macro @code{EWOULDBLOCK} is another name for @code{EAGAIN};
TRANS they are always the same in @theglibc{}.
TRANS
TRANS This error can happen in a few different situations:
TRANS
TRANS @itemize @bullet
TRANS @item
TRANS An operation that would block was attempted on an object that has
TRANS non-blocking mode selected.  Trying the same operation again will block
TRANS until some external condition makes it possible to read, write, or
TRANS connect (whatever the operation).  You can use @code{select} to find out
TRANS when the operation will be possible; @pxref{Waiting for I/O}.
TRANS
TRANS @strong{Portability Note:} In many older Unix systems, this condition
TRANS was indicated by @code{EWOULDBLOCK}, which was a distinct error code
TRANS different from @code{EAGAIN}.  To make your program portable, you should
TRANS check for both codes and treat them the same.
TRANS
TRANS @item
TRANS A temporary resource shortage made an operation impossible.  @code{fork}
TRANS can return this error.  It indicates that the shortage is expected to
TRANS pass, so your program can try the call again later and it may succeed.
TRANS It is probably a good idea to delay for a few seconds before trying it
TRANS again, to allow time for other processes to release scarce resources.
TRANS Such shortages are usually fairly serious and affect the whole system,
TRANS so usually an interactive program should report the error to the user
TRANS and return to its command loop.
TRANS @end itemize */
_S(EAGAIN, N_("Resource temporarily unavailable"))
#endif
#ifdef EINPROGRESS
/*
TRANS An operation that cannot complete immediately was initiated on an object
TRANS that has non-blocking mode selected.  Some functions that must always
TRANS block (such as @code{connect}; @pxref{Connecting}) never return
TRANS @code{EAGAIN}.  Instead, they return @code{EINPROGRESS} to indicate that
TRANS the operation has begun and will take some time.  Attempts to manipulate
TRANS the object before the call completes return @code{EALREADY}.  You can
TRANS use the @code{select} function to find out when the pending operation
TRANS has completed; @pxref{Waiting for I/O}. */
_S(EINPROGRESS, N_("Operation now in progress"))
#endif
#ifdef EALREADY
/*
TRANS An operation is already in progress on an object that has non-blocking
TRANS mode selected. */
_S(EALREADY, N_("Operation already in progress"))
#endif
#ifdef ENOTSOCK
/*
TRANS A file that isn't a socket was specified when a socket is required. */
_S(ENOTSOCK, N_("Socket operation on non-socket"))
#endif
#ifdef EMSGSIZE
/*
TRANS The size of a message sent on a socket was larger than the supported
TRANS maximum size. */
_S(EMSGSIZE, N_("Message too long"))
#endif
#ifdef EPROTOTYPE
/*
TRANS The socket type does not support the requested communications protocol. */
_S(EPROTOTYPE, N_("Protocol wrong type for socket"))
#endif
#ifdef ENOPROTOOPT
/*
TRANS You specified a socket option that doesn't make sense for the
TRANS particular protocol being used by the socket.  @xref{Socket Options}. */
_S(ENOPROTOOPT, N_("Protocol not available"))
#endif
#ifdef EPROTONOSUPPORT
/*
TRANS The socket domain does not support the requested communications protocol
TRANS (perhaps because the requested protocol is completely invalid).
TRANS @xref{Creating a Socket}. */
_S(EPROTONOSUPPORT, N_("Protocol not supported"))
#endif
#ifdef ESOCKTNOSUPPORT
/*
TRANS The socket type is not supported. */
_S(ESOCKTNOSUPPORT, N_("Socket type not supported"))
#endif
#ifdef EOPNOTSUPP
/*
TRANS The operation you requested is not supported.  Some socket functions
TRANS don't make sense for all types of sockets, and others may not be
TRANS implemented for all communications protocols.  On @gnuhurdsystems{}, this
TRANS error can happen for many calls when the object does not support the
TRANS particular operation; it is a generic indication that the server knows
TRANS nothing to do for that call. */
_S(EOPNOTSUPP, N_("Operation not supported"))
#endif
#ifdef EPFNOSUPPORT
/*
TRANS The socket communications protocol family you requested is not supported. */
_S(EPFNOSUPPORT, N_("Protocol family not supported"))
#endif
#ifdef EAFNOSUPPORT
/*
TRANS The address family specified for a socket is not supported; it is
TRANS inconsistent with the protocol being used on the socket.  @xref{Sockets}. */
_S(EAFNOSUPPORT, N_("Address family not supported by protocol"))
#endif
#ifdef EADDRINUSE
/*
TRANS The requested socket address is already in use.  @xref{Socket Addresses}. */
_S(EADDRINUSE, N_("Address already in use"))
#endif
#ifdef EADDRNOTAVAIL
/*
TRANS The requested socket address is not available; for example, you tried
TRANS to give a socket a name that doesn't match the local host name.
TRANS @xref{Socket Addresses}. */
_S(EADDRNOTAVAIL, N_("Cannot assign requested address"))
#endif
#ifdef ENETDOWN
/*
TRANS A socket operation failed because the network was down. */
_S(ENETDOWN, N_("Network is down"))
#endif
#ifdef ENETUNREACH
/*
TRANS A socket operation failed because the subnet containing the remote host
TRANS was unreachable. */
_S(ENETUNREACH, N_("Network is unreachable"))
#endif
#ifdef ENETRESET
/*
TRANS A network connection was reset because the remote host crashed. */
_S(ENETRESET, N_("Network dropped connection on reset"))
#endif
#ifdef ECONNABORTED
/*
TRANS A network connection was aborted locally. */
_S(ECONNABORTED, N_("Software caused connection abort"))
#endif
#ifdef ECONNRESET
/*
TRANS A network connection was closed for reasons outside the control of the
TRANS local host, such as by the remote machine rebooting or an unrecoverable
TRANS protocol violation. */
_S(ECONNRESET, N_("Connection reset by peer"))
#endif
#ifdef ENOBUFS
/*
TRANS The kernel's buffers for I/O operations are all in use.  In GNU, this
TRANS error is always synonymous with @code{ENOMEM}; you may get one or the
TRANS other from network operations. */
_S(ENOBUFS, N_("No buffer space available"))
#endif
#ifdef EISCONN
/*
TRANS You tried to connect a socket that is already connected.
TRANS @xref{Connecting}. */
_S(EISCONN, N_("Transport endpoint is already connected"))
#endif
#ifdef ENOTCONN
/*
TRANS The socket is not connected to anything.  You get this error when you
TRANS try to transmit data over a socket, without first specifying a
TRANS destination for the data.  For a connectionless socket (for datagram
TRANS protocols, such as UDP), you get @code{EDESTADDRREQ} instead. */
_S(ENOTCONN, N_("Transport endpoint is not connected"))
#endif
#ifdef EDESTADDRREQ
/*
TRANS No default destination address was set for the socket.  You get this
TRANS error when you try to transmit data over a connectionless socket,
TRANS without first specifying a destination for the data with @code{connect}. */
_S(EDESTADDRREQ, N_("Destination address required"))
#endif
#ifdef ESHUTDOWN
/*
TRANS The socket has already been shut down. */
_S(ESHUTDOWN, N_("Cannot send after transport endpoint shutdown"))
#endif
#ifdef ETOOMANYREFS
_S(ETOOMANYREFS, N_("Too many references: cannot splice"))
#endif
#ifdef ETIMEDOUT
/*
TRANS A socket operation with a specified timeout received no response during
TRANS the timeout period. */
_S(ETIMEDOUT, N_("Connection timed out"))
#endif
#ifdef ECONNREFUSED
/*
TRANS A remote host refused to allow the network connection (typically because
TRANS it is not running the requested service). */
_S(ECONNREFUSED, N_("Connection refused"))
#endif
#ifdef ELOOP
/*
TRANS Too many levels of symbolic links were encountered in looking up a file name.
TRANS This often indicates a cycle of symbolic links. */
_S(ELOOP, N_("Too many levels of symbolic links"))
#endif
#ifdef ENAMETOOLONG
/*
TRANS Filename too long (longer than @code{PATH_MAX}; @pxref{Limits for
TRANS Files}) or host name too long (in @code{gethostname} or
TRANS @code{sethostname}; @pxref{Host Identification}). */
_S(ENAMETOOLONG, N_("File name too long"))
#endif
#ifdef EHOSTDOWN
/*
TRANS The remote host for a requested network connection is down. */
_S(EHOSTDOWN, N_("Host is down"))
#endif
/*
TRANS The remote host for a requested network connection is not reachable. */
#ifdef EHOSTUNREACH
_S(EHOSTUNREACH, N_("No route to host"))
#endif
#ifdef ENOTEMPTY
/*
TRANS Directory not empty, where an empty directory was expected.  Typically,
TRANS this error occurs when you are trying to delete a directory. */
_S(ENOTEMPTY, N_("Directory not empty"))
#endif
#ifdef EUSERS
/*
TRANS The file quota system is confused because there are too many users.
TRANS @c This can probably happen in a GNU system when using NFS. */
_S(EUSERS, N_("Too many users"))
#endif
#ifdef EDQUOT
/*
TRANS The user's disk quota was exceeded. */
_S(EDQUOT, N_("Disk quota exceeded"))
#endif
#ifdef ESTALE
/*
TRANS This indicates an internal confusion in the
TRANS file system which is due to file system rearrangements on the server host
TRANS for NFS file systems or corruption in other file systems.
TRANS Repairing this condition usually requires unmounting, possibly repairing
TRANS and remounting the file system. */
_S(ESTALE, N_("Stale file handle"))
#endif
#ifdef EREMOTE
/*
TRANS An attempt was made to NFS-mount a remote file system with a file name that
TRANS already specifies an NFS-mounted file.
TRANS (This is an error on some operating systems, but we expect it to work
TRANS properly on @gnuhurdsystems{}, making this error code impossible.) */
_S(EREMOTE, N_("Object is remote"))
#endif
#ifdef ENOLCK
/*
TRANS This is used by the file locking facilities; see
TRANS @ref{File Locks}.  This error is never generated by @gnuhurdsystems{}, but
TRANS it can result from an operation to an NFS server running another
TRANS operating system. */
_S(ENOLCK, N_("No locks available"))
#endif
#ifdef ENOSYS
/*
TRANS This indicates that the function called is
TRANS not implemented at all, either in the C library itself or in the
TRANS operating system.  When you get this error, you can be sure that this
TRANS particular function will always fail with @code{ENOSYS} unless you
TRANS install a new version of the C library or the operating system. */
_S(ENOSYS, N_("Function not implemented"))
#endif
#ifdef EILSEQ
/*
TRANS While decoding a multibyte character the function came along an invalid
TRANS or an incomplete sequence of bytes or the given wide character is invalid. */
_S(EILSEQ, N_("Invalid or incomplete multibyte or wide character"))
#endif
#ifdef EBADMSG
_S(EBADMSG, N_("Bad message"))
#endif
#ifdef EIDRM
_S(EIDRM, N_("Identifier removed"))
#endif
#ifdef EMULTIHOP
_S(EMULTIHOP, N_("Multihop attempted"))
#endif
#ifdef ENODATA
_S(ENODATA, N_("No data available"))
#endif
#ifdef ENOLINK
_S(ENOLINK, N_("Link has been severed"))
#endif
#ifdef ENOMSG
_S(ENOMSG, N_("No message of desired type"))
#endif
#ifdef ENOSR
_S(ENOSR, N_("Out of streams resources"))
#endif
#ifdef ENOSTR
_S(ENOSTR, N_("Device not a stream"))
#endif
#ifdef EOVERFLOW
_S(EOVERFLOW, N_("Value too large for defined data type"))
#endif
#ifdef EPROTO
_S(EPROTO, N_("Protocol error"))
#endif
#ifdef ETIME
_S(ETIME, N_("Timer expired"))
#endif
#ifdef ECANCELED
/*
TRANS An asynchronous operation was canceled before it
TRANS completed.  @xref{Asynchronous I/O}.  When you call @code{aio_cancel},
TRANS the normal result is for the operations affected to complete with this
TRANS error; @pxref{Cancel AIO Operations}. */
_S(ECANCELED, N_("Operation canceled"))
#endif
#ifdef EOWNERDEAD
_S(EOWNERDEAD, N_("Owner died"))
#endif
#ifdef ENOTRECOVERABLE
_S(ENOTRECOVERABLE, N_("State not recoverable"))
#endif
#ifdef ERESTART
_S(ERESTART, N_("Interrupted system call should be restarted"))
#endif
#ifdef ECHRNG
_S(ECHRNG, N_("Channel number out of range"))
#endif
#ifdef EL2NSYNC
_S(EL2NSYNC, N_("Level 2 not synchronized"))
#endif
#ifdef EL3HLT
_S(EL3HLT, N_("Level 3 halted"))
#endif
#ifdef EL3RST
_S(EL3RST, N_("Level 3 reset"))
#endif
#ifdef ELNRNG
_S(ELNRNG, N_("Link number out of range"))
#endif
#ifdef EUNATCH
_S(EUNATCH, N_("Protocol driver not attached"))
#endif
#ifdef ENOCSI
_S(ENOCSI, N_("No CSI structure available"))
#endif
#ifdef EL2HLT
_S(EL2HLT, N_("Level 2 halted"))
#endif
#ifdef EBADE
_S(EBADE, N_("Invalid exchange"))
#endif
#ifdef EBADR
_S(EBADR, N_("Invalid request descriptor"))
#endif
#ifdef EXFULL
_S(EXFULL, N_("Exchange full"))
#endif
#ifdef ENOANO
_S(ENOANO, N_("No anode"))
#endif
#ifdef EBADRQC
_S(EBADRQC, N_("Invalid request code"))
#endif
#ifdef EBADSLT
_S(EBADSLT, N_("Invalid slot"))
#endif
#ifdef EBFONT
_S(EBFONT, N_("Bad font file format"))
#endif
#ifdef ENONET
_S(ENONET, N_("Machine is not on the network"))
#endif
#ifdef ENOPKG
_S(ENOPKG, N_("Package not installed"))
#endif
#ifdef EADV
_S(EADV, N_("Advertise error"))
#endif
#ifdef ESRMNT
_S(ESRMNT, N_("Srmount error"))
#endif
#ifdef ECOMM
_S(ECOMM, N_("Communication error on send"))
#endif
#ifdef EDOTDOT
_S(EDOTDOT, N_("RFS specific error"))
#endif
#ifdef ENOTUNIQ
_S(ENOTUNIQ, N_("Name not unique on network"))
#endif
#ifdef EBADFD
_S(EBADFD, N_("File descriptor in bad state"))
#endif
#ifdef EREMCHG
_S(EREMCHG, N_("Remote address changed"))
#endif
#ifdef ELIBACC
_S(ELIBACC, N_("Can not access a needed shared library"))
#endif
#ifdef ELIBBAD
_S(ELIBBAD, N_("Accessing a corrupted shared library"))
#endif
#ifdef ELIBSCN
_S(ELIBSCN, N_(".lib section in a.out corrupted"))
#endif
#ifdef ELIBMAX
_S(ELIBMAX, N_("Attempting to link in too many shared libraries"))
#endif
#ifdef ELIBEXEC
_S(ELIBEXEC, N_("Cannot exec a shared library directly"))
#endif
#ifdef ESTRPIPE
_S(ESTRPIPE, N_("Streams pipe error"))
#endif
#ifdef EUCLEAN
_S(EUCLEAN, N_("Structure needs cleaning"))
#endif
#ifdef ENOTNAM
_S(ENOTNAM, N_("Not a XENIX named type file"))
#endif
#ifdef ENAVAIL
_S(ENAVAIL, N_("No XENIX semaphores available"))
#endif
#ifdef EISNAM
_S(EISNAM, N_("Is a named type file"))
#endif
#ifdef EREMOTEIO
_S(EREMOTEIO, N_("Remote I/O error"))
#endif
#ifdef ENOMEDIUM
_S(ENOMEDIUM, N_("No medium found"))
#endif
#ifdef EMEDIUMTYPE
_S(EMEDIUMTYPE, N_("Wrong medium type"))
#endif
#ifdef ENOKEY
_S(ENOKEY, N_("Required key not available"))
#endif
#ifdef EKEYEXPIRED
_S(EKEYEXPIRED, N_("Key has expired"))
#endif
#ifdef EKEYREVOKED
_S(EKEYREVOKED, N_("Key has been revoked"))
#endif
#ifdef EKEYREJECTED
_S(EKEYREJECTED, N_("Key was rejected by service"))
#endif
#ifdef ERFKILL
_S(ERFKILL, N_("Operation not possible due to RF-kill"))
#endif
#ifdef EHWPOISON
_S(EHWPOISON, N_("Memory page has hardware error"))
#endif
#ifdef EBADRPC
_S(EBADRPC, N_("RPC struct is bad"))
#endif
#ifdef EFTYPE
/*
TRANS The file was the wrong type for the
TRANS operation, or a data file had the wrong format.
TRANS
TRANS On some systems @code{chmod} returns this error if you try to set the
TRANS sticky bit on a non-directory file; @pxref{Setting Permissions}. */
_S(EFTYPE, N_("Inappropriate file type or format"))
#endif
#ifdef EPROCUNAVAIL
_S(EPROCUNAVAIL, N_("RPC bad procedure for program"))
#endif
#ifdef EAUTH
_S(EAUTH, N_("Authentication error"))
#endif
#ifdef EDIED
/*
TRANS On @gnuhurdsystems{}, opening a file returns this error when the file is
TRANS translated by a program and the translator program dies while starting
TRANS up, before it has connected to the file. */
_S(EDIED, N_("Translator died"))
#endif
#ifdef ERPCMISMATCH
_S(ERPCMISMATCH, N_("RPC version wrong"))
#endif
#ifdef EGREGIOUS
/*
TRANS You did @strong{what}? */
_S(EGREGIOUS, N_("You really blew it this time"))
#endif
#ifdef EPROCLIM
/*
TRANS This means that the per-user limit on new process would be exceeded by
TRANS an attempted @code{fork}.  @xref{Limits on Resources}, for details on
TRANS the @code{RLIMIT_NPROC} limit. */
_S(EPROCLIM, N_("Too many processes"))
#endif
#ifdef EGRATUITOUS
/*
TRANS This error code has no purpose. */
_S(EGRATUITOUS, N_("Gratuitous error"))
#endif
#if defined (ENOTSUP) && ENOTSUP != EOPNOTSUPP
/*
TRANS A function returns this error when certain parameter
TRANS values are valid, but the functionality they request is not available.
TRANS This can mean that the function does not implement a particular command
TRANS or option value or flag bit at all.  For functions that operate on some
TRANS object given in a parameter, such as a file descriptor or a port, it
TRANS might instead mean that only @emph{that specific object} (file
TRANS descriptor, port, etc.) is unable to support the other parameters given;
TRANS different file descriptors might support different ranges of parameter
TRANS values.
TRANS
TRANS If the entire function is not available at all in the implementation,
TRANS it returns @code{ENOSYS} instead. */
_S(ENOTSUP, N_("Not supported"))
#endif
#ifdef EPROGMISMATCH
_S(EPROGMISMATCH, N_("RPC program version wrong"))
#endif
#ifdef EBACKGROUND
/*
TRANS On @gnuhurdsystems{}, servers supporting the @code{term} protocol return
TRANS this error for certain operations when the caller is not in the
TRANS foreground process group of the terminal.  Users do not usually see this
TRANS error because functions such as @code{read} and @code{write} translate
TRANS it into a @code{SIGTTIN} or @code{SIGTTOU} signal.  @xref{Job Control},
TRANS for information on process groups and these signals. */
_S(EBACKGROUND, N_("Inappropriate operation for background process"))
#endif
#ifdef EIEIO
/*
TRANS Go home and have a glass of warm, dairy-fresh milk.
TRANS @c Okay.  Since you are dying to know, I'll tell you.
TRANS @c This is a joke, obviously.  There is a children's song which begins,
TRANS @c "Old McDonald had a farm, e-i-e-i-o."  Every time I see the (real)
TRANS @c errno macro EIO, I think about that song.  Probably most of my
TRANS @c compatriots who program on Unix do, too.  One of them must have stayed
TRANS @c up a little too late one night and decided to add it to Hurd or Glibc.
TRANS @c Whoever did it should be castigated, but it made me laugh.
TRANS @c  --jtobey@channel1.com
TRANS @c
TRANS @c "bought the farm" means "died".  -jtobey
TRANS @c
TRANS @c Translators, please do not translate this litteraly, translate it into
TRANS @c an idiomatic funny way of saying that the computer died. */
_S(EIEIO, N_("Computer bought the farm"))
#endif
#if defined (EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
/*
TRANS In @theglibc{}, this is another name for @code{EAGAIN} (above).
TRANS The values are always the same, on every operating system.
TRANS
TRANS C libraries in many older Unix systems have @code{EWOULDBLOCK} as a
TRANS separate error code. */
_S(EWOULDBLOCK, N_("Operation would block"))
#endif
#ifdef ENEEDAUTH
_S(ENEEDAUTH, N_("Need authenticator"))
#endif
#ifdef ED
/*
TRANS The experienced user will know what is wrong.
TRANS @c This error code is a joke.  Its perror text is part of the joke.
TRANS @c Don't change it. */
_S(ED, N_("?"))
#endif
#ifdef EPROGUNAVAIL
_S(EPROGUNAVAIL, N_("RPC program not available"))
#endif
