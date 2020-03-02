def StdC : StandardSpec<"stdc"> {
  ConstType ConstVoidPtr = ConstType<VoidPtr>;
  RestrictedPtrType VoidRestrictedPtr = RestrictedPtrType<VoidType>;
  ConstType ConstVoidRestrictedPtr = ConstType<VoidRestrictedPtr>;

  PtrType CharPtr = PtrType<CharType>;
  ConstType ConstCharPtr = ConstType<CharPtr>;
  RestrictedPtrType CharRestrictedPtr = RestrictedPtrType<CharType>;
  ConstType ConstCharRestrictedPtr = ConstType<CharRestrictedPtr>;

  NamedType MtxTType = NamedType<"mtx_t">;
  PtrType MtxTTypePtr = PtrType<MtxTType>;
  NamedType ThrdStartTType = NamedType<"thrd_start_t">;
  NamedType ThrdTType = NamedType<"thrd_t">;
  PtrType ThrdTTypePtr = PtrType<ThrdTType>;

  PtrType IntPtr = PtrType<IntType>;

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
              "memcpy",
              RetValSpec<VoidPtr>, 
              [ArgSpec<VoidRestrictedPtr>,
               ArgSpec<ConstVoidRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memmove",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memcmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memchr",
              RetValSpec<VoidPtr>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memset",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcpy",
              RetValSpec<CharPtr>, 
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strncpy",
              RetValSpec<CharPtr>, 
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcat",
              RetValSpec<CharPtr>, 
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strncat",
              RetValSpec<CharPtr>, 
              [ArgSpec<CharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strcoll",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strncmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strxfrm",
              RetValSpec<SizeTType>, 
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strchr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strcspn",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strpbrk",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strrchr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strspn",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strstr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strtok",
              RetValSpec<CharPtr>,
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strerror",
              RetValSpec<CharPtr>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strlen",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>]
          >,
      ]
  >;

  HeaderSpec Math = HeaderSpec<
      "math.h",
      [], // Macros
      [
          NamedType<"float_t">,
          NamedType<"double_t">,
      ],
      [], // Enumerations
      [
          FunctionSpec<"acos", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"acosl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
      ]
  >;

  HeaderSpec StdIO = HeaderSpec<
      "stdio.h",
      [], // Macros
      [ // Types
          SizeTType,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "snprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
      ]
  >;

  HeaderSpec Errno = HeaderSpec<
      "errno.h",
      [
        Macro<"errno">,
        Macro<"EDOM">,
        Macro<"EILSEQ">,
        Macro<"ERANGE">,
      ]
  >;

  HeaderSpec Signal = HeaderSpec<
      "signal.h",
      [
        Macro<"SIG_BLOCK">,
        Macro<"SIG_UNBLOCK">,
        Macro<"SIG_SETMASK">,

        Macro<"SIGABRT">,
        Macro<"SIGFPE">,
        Macro<"SIGILL">,
        Macro<"SIGINT">,
        Macro<"SIGSEGV">,
        Macro<"SIGTERM">
      ],
      [
        NamedType<"sigset_t">,
        SizeTType,
      ],
      [], // Enumerations
      [
        FunctionSpec<"raise", RetValSpec<IntType>, [ArgSpec<IntType>]>,
      ]
  >;

  HeaderSpec Threads = HeaderSpec<
      "threads.h",
      [], // Macros
      [
          MtxTType,
          ThrdStartTType,
          ThrdTType,
      ],
      [
          EnumeratedNameValue<"mtx_plain">,
          EnumeratedNameValue<"mtx_recursive">,
          EnumeratedNameValue<"mtx_timed">,
          EnumeratedNameValue<"thrd_timedout">,
          EnumeratedNameValue<"thrd_success">,
          EnumeratedNameValue<"thrd_busy">,
          EnumeratedNameValue<"thrd_error">,
          EnumeratedNameValue<"thrd_nomem">,
      ],
      [
          FunctionSpec<
              "mtx_init",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
                  ArgSpec<IntType>,
              ]
          >,
          FunctionSpec<
              "mtx_lock",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
              ]
          >,
          FunctionSpec<
              "mtx_unlock",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
              ]
          >,
          FunctionSpec<
              "thrd_create",
              RetValSpec<IntType>,
              [
                  ArgSpec<ThrdTTypePtr>,
                  ArgSpec<ThrdStartTType>,
                  ArgSpec<VoidPtr>,
              ]
          >,
          FunctionSpec<
              "thrd_join",
              RetValSpec<IntType>,
              [
                  ArgSpec<ThrdTTypePtr>,
                  ArgSpec<IntPtr>,
              ]
          >
      ]
  >;

  let Headers = [
    Errno,
    Math,
    String,
    StdIO,
    Signal,
    Threads,
  ];
}
