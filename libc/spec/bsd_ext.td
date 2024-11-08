def BsdExtensions : StandardSpec<"BSDExtensions"> {
  HeaderSpec Math = HeaderSpec<
      "math.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<"isnan", RetValSpec<IntType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"isnanf", RetValSpec<IntType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"isnanl", RetValSpec<IntType>, [ArgSpec<LongDoubleType>]>,
      ]
  >;

  HeaderSpec String = HeaderSpec<
      "string.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "strlcat",
            RetValSpec<SizeTType>,
            [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "strlcpy",
            RetValSpec<SizeTType>,
            [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "strsep",
            RetValSpec<CharPtr>,
            [ArgSpec<CharRestrictedPtrPtr>, ArgSpec<ConstCharRestrictedPtr>]
        >,
      ]
  >;

  HeaderSpec Strings = HeaderSpec<
      "strings.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "strcasecmp",
            RetValSpec<IntType>,
            [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
        >,
        FunctionSpec<
            "strncasecmp",
            RetValSpec<IntType>,
            [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "index",
            RetValSpec<CharPtr>,
            [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
        >,
        FunctionSpec<
            "rindex",
            RetValSpec<CharPtr>,
            [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
        >,
      ]
  >;

  HeaderSpec SysWait = HeaderSpec<
      "sys/wait.h",
      [], // Macros
      [StructRUsage], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "wait4",
            RetValSpec<PidT>,
            [ArgSpec<PidT>, ArgSpec<IntPtr>, ArgSpec<IntType>, ArgSpec<StructRUsagePtr>]
        >
      ]
  >;

  let Headers = [
    Math,
    String,
    Strings,
    SysWait,
  ];
}
