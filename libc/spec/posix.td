def SigSetType : NamedType<"sigset_t">;
def SigSetPtrType : PtrType<SigSetType>;
def ConstSigSetPtrType : ConstType<SigSetPtrType>;
def RestrictedSigSetType : RestrictedPtrType<SigSetType>;
def ConstRestrictedSigSetType : ConstType<RestrictedSigSetType>;

def StructSigaction : NamedType<"struct sigaction">;
def StructSigactionPtr : PtrType<StructSigaction>;
def ConstStructSigactionPtr : ConstType<StructSigactionPtr>;
def RestrictedStructSigactionPtr : RestrictedPtrType<StructSigaction>;
def ConstRestrictedStructSigactionPtr : ConstType<RestrictedStructSigactionPtr>;

def POSIX : StandardSpec<"POSIX"> {
  PtrType CharPtr = PtrType<CharType>;
  RestrictedPtrType RestrictedCharPtr = RestrictedPtrType<CharType>;
  RestrictedPtrType CharRestrictedDoublePtr = RestrictedPtrType<CharPtr>;
  ConstType ConstCharPtr = ConstType<CharPtr>;
  ConstType ConstRestrictedCharPtr = ConstType<RestrictedCharPtr>;

  NamedType ModeTType = NamedType<"mode_t">;
  NamedType OffTType = NamedType<"off_t">;
  NamedType SSizeTType = NamedType<"ssize_t">;

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

  HeaderSpec FCntl = HeaderSpec<
    "fcntl.h",
    [], // Macros
    [ModeTType],
    [], // Enumerations
    [
      FunctionSpec<
          "creat",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<ModeTType>]
      >,
      FunctionSpec<
          "open",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<IntType>, ArgSpec<VarArgType>]
      >,
      FunctionSpec<
          "openat",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<IntType>, ArgSpec<VarArgType>]
      >,
    ]
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
      [
        SigSetType,
        StructSigaction,
      ],
      [], // Enumerations
      [
        FunctionSpec<
          "sigaction",
          RetValSpec<IntType>,
          [ArgSpec<IntType>,
           ArgSpec<ConstRestrictedStructSigactionPtr>,
           ArgSpec<RestrictedStructSigactionPtr>]
        >,
        FunctionSpec<
          "sigdelset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>,
           ArgSpec<IntType>]
        >,
        FunctionSpec<
          "sigprocmask",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstRestrictedSigSetType>, ArgSpec<RestrictedSigSetType>]
        >,
        FunctionSpec<
          "sigemptyset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>]
        >,
        FunctionSpec<
          "sigaddset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>,
           ArgSpec<IntType>]
        >,
        FunctionSpec<
          "sigfillset",
          RetValSpec<IntType>,
          [ArgSpec<SigSetPtrType>]
        >,
      ]
  >;

  HeaderSpec UniStd = HeaderSpec<
    "unistd.h",
    [], // Macros
    [
      SSizeTType,
      SizeTType,
    ],
    [], // Enumerations
    [
        FunctionSpec<
          "close",
          RetValSpec<IntType>,
          [ArgSpec<IntType>]
        >,
        FunctionSpec<
          "fsync",
          RetValSpec<IntType>,
          [ArgSpec<IntType>]
        >,
        FunctionSpec<
          "read",
          RetValSpec<SSizeTType>,
          [ArgSpec<IntType>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
          "rmdir",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "unlink",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "unlinkat",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "write",
          RetValSpec<SSizeTType>,
          [ArgSpec<IntType>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
        >,
    ]
  >;
  
  HeaderSpec String = HeaderSpec<
    "string.h",
    [
        Macro<"NULL">,
    ],
    [
        SizeTType,
    ],
    [], // Enumerations
    [
        FunctionSpec<
            "memccpy",
            RetValSpec<VoidPtr>,
            [ArgSpec<VoidRestrictedPtr>,
             ArgSpec<ConstVoidRestrictedPtr>,
             ArgSpec<IntType>,
             ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "mempcpy",
            RetValSpec<VoidPtr>,
            [ArgSpec<VoidRestrictedPtr>,
             ArgSpec<ConstVoidRestrictedPtr>,
             ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "stpcpy",
            RetValSpec<CharPtr>,
            [ArgSpec<RestrictedCharPtr>,
             ArgSpec<ConstRestrictedCharPtr>]
        >,
        FunctionSpec<
            "stpncpy",
            RetValSpec<CharPtr>,
            [ArgSpec<RestrictedCharPtr>,
             ArgSpec<ConstRestrictedCharPtr>,
             ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "strnlen",
             RetValSpec<SizeTType>,
             [ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "strtok_r",
            RetValSpec<CharPtr>,
            [ArgSpec<RestrictedCharPtr>,
             ArgSpec<ConstRestrictedCharPtr>,
             ArgSpec<CharRestrictedDoublePtr>]
        >,
    ]
  >;

  HeaderSpec CType = HeaderSpec<
    "ctype.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "isascii",
            RetValSpec<IntType>,
            [ArgSpec<IntType>]
        >,
    ]
  >;

  HeaderSpec SysStat = HeaderSpec<
    "sys/stat.h",
    [], // Macros
    [ModeTType], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "mkdir",
            RetValSpec<IntType>,
            [ArgSpec<ConstCharPtr>, ArgSpec<ModeTType>]
        >,
        FunctionSpec<
            "mkdirat",
            RetValSpec<IntType>,
            [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<ModeTType>]
        >,
    ]
  >;

  let Headers = [
    CType,
    Errno,
    FCntl,
    Signal,
    SysMMan,
    SysStat,
    UniStd,
    String
  ];
}
