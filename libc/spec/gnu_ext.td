def GnuExtensions : StandardSpec<"GNUExtensions"> {
  NamedType CookieIOFunctionsT = NamedType<"cookie_io_functions_t">;
  HeaderSpec CType = HeaderSpec<
    "ctype.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "toascii",
            RetValSpec<IntType>,
            [ArgSpec<IntType>]
        >,
    ]
  >;

  HeaderSpec Math = HeaderSpec<
      "math.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "sincosf",
            RetValSpec<VoidType>,
            [ArgSpec<FloatType>, ArgSpec<FloatPtr>, ArgSpec<FloatPtr>]
        >,
      ]
  >;
  
  HeaderSpec String = HeaderSpec<
      "string.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [ 
        FunctionSpec<
            "memrchr",
            RetValSpec<VoidPtr>,
            [ArgSpec<VoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
        >,
      ]
  >;

  HeaderSpec FEnv = HeaderSpec<
      "fenv.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "fedisableexcept",
            RetValSpec<IntType>,
            [ArgSpec<IntType>]
        >,
        FunctionSpec<
            "feenableexcept",
            RetValSpec<IntType>,
            [ArgSpec<IntType>]
        >,
        FunctionSpec<
            "fegetexcept",
            RetValSpec<IntType>,
            []
        >
      ]
  >;

  HeaderSpec StdIO = HeaderSpec<
      "stdio.h",
      [], // Macros
      [CookieIOFunctionsT], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "clearerr_unlocked",
              RetValSpec<VoidType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "feof_unlocked",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "ferror_unlocked",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "fopencookie",
              RetValSpec<FILEPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<ConstCharPtr>, ArgSpec<CookieIOFunctionsT>]
          >,
          FunctionSpec<
              "fread_unlocked",
              RetValSpec<SizeTType>,
              [ArgSpec<VoidRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<SizeTType>,
               ArgSpec<FILERestrictedPtr>]
          >,
          FunctionSpec<
              "fwrite_unlocked",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstVoidRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<SizeTType>,
               ArgSpec<FILERestrictedPtr>]
          >,
      ]
  >;

  HeaderSpec PThread = HeaderSpec<
      "pthread.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "pthread_setname_np",
              RetValSpec<IntType>,
              [ArgSpec<PThreadTType>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "pthread_getname_np",
              RetValSpec<IntType>,
              [ArgSpec<PThreadTType>, ArgSpec<CharPtr>, ArgSpec<SizeTType>]
          >,
      ]
  >;

  let Headers = [
    CType,
    FEnv,
    Math,
    PThread,
    StdIO,
    String,
  ];
}
