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

def PThreadStartT : NamedType<"__pthread_start_t">;
def PThreadTSSDtorT : NamedType<"__pthread_tss_dtor_t">;
def PThreadKeyT : NamedType<"pthread_key_t">;
def PThreadKeyTPtr : PtrType<PThreadKeyT>;

def InoT : NamedType<"ino_t">;
def DIR : NamedType<"DIR">;
def DIRPtr : PtrType<DIR>;
def DIRRestrictedPtr : RestrictedPtrType<DIR>;
def StructDirent : NamedType<"struct dirent">;
def StructDirentPtr : PtrType<StructDirent>;
def StructDirentPtrPtr : PtrType<StructDirentPtr>;
def ConstStructDirentPtrPtr : ConstType<StructDirentPtrPtr>;

def POSIX : StandardSpec<"POSIX"> {
  PtrType CharPtr = PtrType<CharType>;
  RestrictedPtrType RestrictedCharPtr = RestrictedPtrType<CharType>;
  RestrictedPtrType CharRestrictedDoublePtr = RestrictedPtrType<CharPtr>;
  ConstType ConstCharPtr = ConstType<CharPtr>;
  ConstType ConstRestrictedCharPtr = ConstType<RestrictedCharPtr>;

  NamedType ModeTType = NamedType<"mode_t">;

  NamedType PThreadAttrTType = NamedType<"pthread_attr_t">;
  PtrType PThreadAttrTPtr = PtrType<PThreadAttrTType>;
  RestrictedPtrType RestrictedPThreadAttrTPtr = RestrictedPtrType<PThreadAttrTType>;
  ConstType ConstPThreadAttrTPtr = ConstType<PThreadAttrTPtr>;
  ConstType ConstRestrictedPThreadAttrTPtr = ConstType<RestrictedPThreadAttrTPtr>;

  NamedType PThreadMutexAttrTType = NamedType<"pthread_mutexattr_t">;
  PtrType PThreadMutexAttrTPtr = PtrType<PThreadMutexAttrTType>;
  RestrictedPtrType RestrictedPThreadMutexAttrTPtr = RestrictedPtrType<PThreadMutexAttrTType>;
  ConstType ConstPThreadMutexAttrTPtr = ConstType<PThreadMutexAttrTPtr>;
  ConstType ConstRestrictedPThreadMutexAttrTPtr = ConstType<RestrictedPThreadMutexAttrTPtr>;

  NamedType PThreadMutexTType = NamedType<"pthread_mutex_t">;
  PtrType PThreadMutexTPtr = PtrType<PThreadMutexTType>;
  RestrictedPtrType RestrictedPThreadMutexTPtr = RestrictedPtrType<PThreadMutexTType>;
  ConstType ConstPThreadMutexTPtr = ConstType<PThreadMutexTPtr>;
  ConstType ConstRestrictedPThreadMutexTPtr = ConstType<RestrictedPThreadMutexTPtr>;

  PtrType PThreadTPtr = PtrType<PThreadTType>;
  RestrictedPtrType RestrictedPThreadTPtr = RestrictedPtrType<PThreadTType>;

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
      OffTType,
      SSizeTType,
      SizeTType,
    ],
    [], // Enumerations
    [
        FunctionSpec<
          "access",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "chdir",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "dup",
          RetValSpec<IntType>,
          [ArgSpec<IntType>]
        >,
        FunctionSpec<
          "dup2",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "dup3",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<IntType>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "fchdir",
          RetValSpec<IntType>,
          [ArgSpec<IntType>]
        >,
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
          "ftruncate",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<OffTType>]
        >,
        FunctionSpec<
          "link",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "linkat",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "lseek",
          RetValSpec<OffTType>,
          [ArgSpec<IntType>, ArgSpec<OffTType>, ArgSpec<IntType>]
        >,
        FunctionSpec<
          "pread",
          RetValSpec<SSizeTType>,
          [ArgSpec<IntType>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>, ArgSpec<OffTType>]
        >,
        FunctionSpec<
          "pwrite",
          RetValSpec<SSizeTType>,
          [ArgSpec<IntType>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>, ArgSpec<OffTType>]
        >,
        FunctionSpec<
          "read",
          RetValSpec<SSizeTType>,
          [ArgSpec<IntType>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
          "readlink",
          RetValSpec<SSizeTType>,
          [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
          "readlinkat",
          RetValSpec<SSizeTType>,
          [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
          "rmdir",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "symlink",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "symlinkat",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<IntType>, ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
          "truncate",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<OffTType>]
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
  
  HeaderSpec StdLib = HeaderSpec<
    "stdlib.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
          "getenv",
          RetValSpec<CharPtr>,
          [ArgSpec<ConstCharPtr>]
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

  NamedType RLimTType = NamedType<"rlim_t">;
  NamedType StructRLimitTType = NamedType<"struct rlimit_t">;
  PtrType StructRLimitPtr = PtrType<StructRLimitTType>;
  ConstType ConstStructRLimitPtr = ConstType<StructRLimitTType>;
  HeaderSpec SysResource = HeaderSpec<
    "sys/resource.h",
    [], // Macros
    [RLimTType, StructRLimitTType], // Types
    [], // Enumerations
    [
        FunctionSpec<
          "getrlimit",
          RetValSpec<IntType>,
          [ArgSpec<StructRLimitPtr>]
        >,
        FunctionSpec<
          "setrlimit",
          RetValSpec<IntType>,
          [ArgSpec<ConstStructRLimitPtr>]
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
          "chmod",
          RetValSpec<IntType>,
          [ArgSpec<ConstCharPtr>, ArgSpec<ModeTType>]
        >,
        FunctionSpec<
          "fchmod",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ModeTType>]
        >,
        FunctionSpec<
          "fchmodat",
          RetValSpec<IntType>,
          [ArgSpec<IntType>, ArgSpec<ConstCharPtr>, ArgSpec<ModeTType>, ArgSpec<IntType>]
        >,
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

  HeaderSpec PThread = HeaderSpec<
    "pthread.h",
    [], // Macros
    [
        PThreadAttrTType,
        PThreadKeyT,
        PThreadMutexAttrTType,
        PThreadMutexTType,
        PThreadStartT,
        PThreadTSSDtorT,
        PThreadTType,
    ], // Types
    [], // Enumerations
    [
      FunctionSpec<
          "pthread_attr_init",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>]
      >,
      FunctionSpec<
          "pthread_attr_destroy",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>]
      >,
      FunctionSpec<
          "pthread_attr_getdetachstate",
          RetValSpec<IntType>,
          [ArgSpec<ConstPThreadAttrTPtr>, ArgSpec<IntPtr>]
      >,
      FunctionSpec<
          "pthread_attr_setdetachstate",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_attr_getguardsize",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadAttrTPtr>, ArgSpec<RestrictedSizeTPtr>]
      >,
      FunctionSpec<
          "pthread_attr_setguardsize",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>, ArgSpec<SizeTType>]
      >,
      FunctionSpec<
          "pthread_attr_getstacksize",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadAttrTPtr>, ArgSpec<RestrictedSizeTPtr>]
      >,
      FunctionSpec<
          "pthread_attr_setstacksize",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>, ArgSpec<SizeTType>]
      >,
      FunctionSpec<
          "pthread_attr_getstack",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadAttrTPtr>, ArgSpec<RestrictedVoidPtrPtr>, ArgSpec<RestrictedSizeTPtr>]
      >,
      FunctionSpec<
          "pthread_attr_setstack",
          RetValSpec<IntType>,
          [ArgSpec<PThreadAttrTPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
      >,
      FunctionSpec<
          "pthread_create",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadTPtr>, ArgSpec<ConstRestrictedPThreadAttrTPtr>, ArgSpec<PThreadStartT>, ArgSpec<VoidPtr>]
      >,
      FunctionSpec<
          "pthread_join",
          RetValSpec<IntType>,
          [ArgSpec<PThreadTType>, ArgSpec<VoidPtrPtr>]
      >,
      FunctionSpec<
          "pthread_detach",
          RetValSpec<IntType>,
          [ArgSpec<PThreadTType>]
      >,
      FunctionSpec<
          "pthread_exit",
          RetValSpec<VoidType>,
          [ArgSpec<VoidPtr>]
      >,
      FunctionSpec<
          "pthread_self",
          RetValSpec<PThreadTType>,
          [ArgSpec<VoidType>]
      >,
      FunctionSpec<
          "pthread_equal",
          RetValSpec<IntType>,
          [ArgSpec<PThreadTType>, ArgSpec<PThreadTType>]
      >,
      FunctionSpec<
          "pthread_mutexattr_init",
          RetValSpec<IntType>,
          [ArgSpec<PThreadMutexAttrTPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_destroy",
          RetValSpec<IntType>,
          [ArgSpec<PThreadMutexAttrTPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_gettype",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>, ArgSpec<RestrictedIntPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_settype",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_mutexattr_getrobust",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>, ArgSpec<RestrictedIntPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_setrobust",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_mutexattr_getpshared",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>, ArgSpec<RestrictedIntPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_setpshared",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_mutexattr_getprotocol",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>, ArgSpec<RestrictedIntPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_setprotocol",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_mutexattr_getprioceiling",
          RetValSpec<IntType>,
          [ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>, ArgSpec<RestrictedIntPtr>]
      >,
      FunctionSpec<
          "pthread_mutexattr_setprioceiling",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexAttrTPtr>, ArgSpec<IntType>]
      >,
      FunctionSpec<
          "pthread_mutex_init",
          RetValSpec<IntType>,
          [ArgSpec<RestrictedPThreadMutexTPtr>, ArgSpec<ConstRestrictedPThreadMutexAttrTPtr>]
      >,
      FunctionSpec<
          "pthread_mutex_destroy",
          RetValSpec<IntType>,
          [ArgSpec<PThreadMutexTPtr>]
      >,
      FunctionSpec<
          "pthread_mutex_lock",
          RetValSpec<IntType>,
          [ArgSpec<PThreadMutexTPtr>]
      >,
      FunctionSpec<
          "pthread_mutex_unlock",
          RetValSpec<IntType>,
          [ArgSpec<PThreadMutexTPtr>]
      >,
      FunctionSpec<
          "pthread_key_create",
          RetValSpec<IntType>,
          [ArgSpec<PThreadKeyTPtr>, ArgSpec<PThreadTSSDtorT>]
      >,
      FunctionSpec<
          "pthread_key_delete",
          RetValSpec<IntType>,
          [ArgSpec<PThreadKeyT>]
      >,
      FunctionSpec<
          "pthread_getspecific",
          RetValSpec<VoidPtr>,
          [ArgSpec<PThreadKeyT>]
      >,
      FunctionSpec<
          "pthread_setspecific",
          RetValSpec<VoidPtr>,
          [ArgSpec<PThreadKeyT>, ArgSpec<ConstVoidPtr>]
      >,
    ]
  >;

  HeaderSpec StdIO = HeaderSpec<
      "stdio.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "flockfile",
              RetValSpec<VoidType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "funlockfile",
              RetValSpec<VoidType>,
              [ArgSpec<FILEPtr>]
          >,
      ]
  >;

  HeaderSpec Dirent = HeaderSpec<
      "dirent.h",
      [], // Macros
      [InoT, StructDirent, DIR], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "alphasort",
              RetValSpec<IntType>,
              [ArgSpec<ConstStructDirentPtrPtr>, ArgSpec<ConstStructDirentPtrPtr>]
          >,
          FunctionSpec<
              "closedir",
              RetValSpec<IntType>,
              [ArgSpec<DIRPtr>]
          >,
          FunctionSpec<
              "dirfd",
              RetValSpec<IntType>,
              [ArgSpec<DIRPtr>]
          >,
          FunctionSpec<
              "fdopendir",
              RetValSpec<DIRPtr>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "opendir",
              RetValSpec<DIRPtr>,
              [ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "readdir",
              RetValSpec<StructDirentPtr>,
              [ArgSpec<DIRPtr>]
          >,
      ]
  >;

  let Headers = [
    CType,
    Dirent,
    Errno,
    FCntl,
    PThread,
    Signal,
    StdIO,
    StdLib,
    SysMMan,
    SysResource,
    SysStat,
    UniStd,
    String
  ];
}
