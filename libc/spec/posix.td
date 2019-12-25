def POSIX : StandardSpec<"POSIX"> {
  NamedType OffTType = NamedType<"off_t">;

  HeaderSpec Errno = HeaderSpec<
      "errno.h",
      [
        Macro<"E2BIG">,
        Macro<"EACCES">,
        Macro<"EADDRINUSE">,
        Macro<"EADDRNOTAVAIL">,
        Macro<"EAFNOSUPPORT">,
        Macro<"EAGAIN">,
        Macro<"EALREADY">,
        Macro<"EBADF">,
        Macro<"EBADMSG">,
        Macro<"EBUSY">,
        Macro<"ECANCELED">,
        Macro<"ECHILD">,
        Macro<"ECONNABORTED">,
        Macro<"ECONNREFUSED">,
        Macro<"ECONNRESET">,
        Macro<"EDEADLK">,
        Macro<"EDESTADDRREQ">,
        Macro<"EDQUOT">,
        Macro<"EEXIST">,
        Macro<"EFAULT">,
        Macro<"EFBIG">,
        Macro<"EHOSTUNREACH">,
        Macro<"EIDRM">,
        Macro<"EINPROGRESS">,
        Macro<"EINTR">,
        Macro<"EINVAL">,
        Macro<"EIO">,
        Macro<"EISCONN">,
        Macro<"EISDIR">,
        Macro<"ELOOP">,
        Macro<"EMFILE">,
        Macro<"EMLINK">,
        Macro<"EMSGSIZE">,
        Macro<"EMULTIHOP">,
        Macro<"ENAMETOOLONG">,
        Macro<"ENETDOWN">,
        Macro<"ENETRESET">,
        Macro<"ENETUNREACH">,
        Macro<"ENFILE">,
        Macro<"ENOBUFS">,
        Macro<"ENODATA">,
        Macro<"ENODEV">,
        Macro<"ENOENT">,
        Macro<"ENOEXEC">,
        Macro<"ENOLCK">,
        Macro<"ENOLINK">,
        Macro<"ENOMEM">,
        Macro<"ENOMSG">,
        Macro<"ENOPROTOOPT">,
        Macro<"ENOSPC">,
        Macro<"ENOSR">,
        Macro<"ENOSTR">,
        Macro<"ENOSYS">,
        Macro<"ENOTCONN">,
        Macro<"ENOTDIR">,
        Macro<"ENOTEMPTY">,
        Macro<"ENOTRECOVERABLE">,
        Macro<"ENOTSOCK">,
        Macro<"ENOTSUP">,
        Macro<"ENOTTY">,
        Macro<"ENXIO">,
        Macro<"EOPNOTSUPP">,
        Macro<"EOVERFLOW">,
        Macro<"EOWNERDEAD">,
        Macro<"EPERM">,
        Macro<"EPIPE">,
        Macro<"EPROTO">,
        Macro<"EPROTONOSUPPORT">,
        Macro<"EPROTOTYPE">,
        Macro<"EROFS">,
        Macro<"ESPIPE">,
        Macro<"ESRCH">,
        Macro<"ESTALE">,
        Macro<"ETIME">,
        Macro<"ETIMEDOUT">,
        Macro<"ETXTBSY">,
        Macro<"EWOULDBLOCK">,
        Macro<"EXDEV">,
      ],
      [], // Types
      []  // Functions
  >;

  HeaderSpec SysMMan = HeaderSpec<
      "sys/mman.h",
      [
        // TODO: Add a facility to bunch macros into bitwise-or-able groups.
        // POSIX requires it, so such thing should be captured in this spec.
        Macro<"PROT_EXEC">,
        Macro<"PROT_NONE">,
        Macro<"PROT_READ">,
        Macro<"PROT_WRITE">,

        Macro<"MAP_FIXED">,
        Macro<"MAP_PRIVATE">,
        Macro<"MAP_SHARED">,

        Macro<"MAP_FAILED">,
      ],
      [
        SizeTType,
        OffTType,
      ],
      [
        FunctionSpec<
            "mmap",
            RetValSpec<VoidPtr>,
            [ArgSpec<VoidPtr>,
             ArgSpec<SizeTType>,
             ArgSpec<IntType>,
             ArgSpec<IntType>,
             ArgSpec<IntType>,
             ArgSpec<OffTType>]
        >,
        FunctionSpec<
            "munmap",
            RetValSpec<IntType>,
            [ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
      ]
  >;

  let Headers = [
    Errno,
    SysMMan,
  ];
}
