def GnuExtensions : StandardSpec<"GNUExtensions"> {
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

  let Headers = [
    CType,
    FEnv,
    Math, 
    String,
  ];
}
