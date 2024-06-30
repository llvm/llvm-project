def LLVMLibcExt : StandardSpec<"llvm_libc_ext"> {
  HeaderSpec Strings = HeaderSpec<
      "strings.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "bcopy",
              RetValSpec<VoidType>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
          >,
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

  HeaderSpec Sched = HeaderSpec<
      "sched.h",
      [], // Macros
      [PidT, SizeTType, CpuSetT], // Types
      [], // Enumerations
      [
        FunctionSpec<
            "__sched_getcpucount",
            RetValSpec<IntType>,
            [ArgSpec<SizeTType>, ArgSpec<ConstCpuSetPtr>]
        >,
      ]
  >;

  HeaderSpec Math = HeaderSpec<
      "math.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          GuardedFunctionSpec<"f16div", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16divf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16divl", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
      ]
  >;

  let Headers = [
    Assert,
    Math,
    Sched,
    Strings,
  ];
}
