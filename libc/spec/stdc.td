def StdC : StandardSpec<"stdc"> {
  ConstType ConstVoidPtr = ConstType<VoidPtr>;
  RestrictedPtrType VoidRestrictedPtr = RestrictedPtrType<VoidType>;
  ConstType ConstVoidRestrictedPtr = ConstType<VoidRestrictedPtr>;

  PtrType CharPtr = PtrType<CharType>;
  ConstType ConstCharPtr = ConstType<CharPtr>;
  RestrictedPtrType CharRestrictedPtr = RestrictedPtrType<CharType>;
  ConstType ConstCharRestrictedPtr = ConstType<CharRestrictedPtr>;

  HeaderSpec String = HeaderSpec<
      "string.h",
      [
          Macro<"NULL">,
      ],
      [
          SizeTType,
      ],
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
      ],
      [], // Types
      []  // Functions
  >;

  let Headers = [
    Errno,
    Math,
    String,
    StdIO,
  ];
}
