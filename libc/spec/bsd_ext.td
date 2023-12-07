def BsdExtensions : StandardSpec<"BSDExtensions"> {
  HeaderSpec String = HeaderSpec<
      "string.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [ 
        FunctionSpec<
            "strlcat",
            RetValSpec<SizeTType>,
            [ArgSpec<CharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "strlcpy",
            RetValSpec<SizeTType>,
            [ArgSpec<CharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
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
    String,
    Strings,
    SysWait,
  ];
}
