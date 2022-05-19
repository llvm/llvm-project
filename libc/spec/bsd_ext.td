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

  let Headers = [
    String,
  ];
}
