def LLVMLibcExt : StandardSpec<"llvm_libc_ext"> {
  HeaderSpec String = HeaderSpec<
      "string.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "bzero",
              RetValSpec<VoidType>,
              [ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "bcmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
          >,
      ]
  >;

  HeaderSpec Assert = HeaderSpec<
      "assert.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "__assert_fail",
              RetValSpec<NoReturn>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<UnsignedType>, ArgSpec<ConstCharPtr>]

          >,
      ]
  >;

  HeaderSpec Errno = HeaderSpec<
      "errno.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "__errno_location",
              RetValSpec<IntPtr>,
              [ArgSpec<VoidType>]

          >,
      ]
  >;

  let Headers = [
    String,
    Assert,
    Errno,
  ];
}
