def BsdExtensions : StandardSpec<"BSDExtensions"> {
  HeaderSpec String = HeaderSpec<
      "string.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [ 
        FunctionSpec<
            "strlcpy",
            RetValSpec<SizeTType>,
            [ArgSpec<CharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
        >,
      ]
  >;
}
