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
    SysWait,
  ];
}
