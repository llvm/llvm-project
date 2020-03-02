def SigSetType : NamedType<"sigset_t">;
def SigSetPtrType : PtrType<SigSetType>;
def ConstSigSetPtrType : ConstType<SigSetPtrType>;
def RestrictSigSetType : RestrictedPtrType<SigSetType>;
def ConstRestrictSigSetType : ConstType<RestrictSigSetType>;

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
      [], // Enumerations
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
      [], // Enumerations
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

  HeaderSpec Signal = HeaderSpec<
      "signal.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
        FunctionSpec<
          "sigprocmask",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstRestrictSigSetType>, ArgSpec<RestrictSigSetType>]
        >,
        FunctionSpec<
          "sigemptyset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>]
        >,
        FunctionSpec<
          "sigaddset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>]
        >,
      ]
  >;

  let Headers = [
    Errno,
    SysMMan,
    Signal,
  ];
}
