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


  let Headers = [
    CType,
    Math, 
    String,
  ];
}
