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

  HeaderSpec StdFix = HeaderSpec<
      "stdfix.h",
      [],  // macros
      [],  // types
      [],  // enums
      [    // functions
          GuardedFunctionSpec<"sqrtuhr", RetValSpec<UnsignedShortFractType>, [ArgSpec<UnsignedShortFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtur", RetValSpec<UnsignedFractType>, [ArgSpec<UnsignedFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtulr", RetValSpec<UnsignedLongFractType>, [ArgSpec<UnsignedLongFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"sqrtuhk", RetValSpec<UnsignedShortAccumType>, [ArgSpec<UnsignedShortAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtuk", RetValSpec<UnsignedAccumType>, [ArgSpec<UnsignedAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtulk", RetValSpec<UnsignedLongAccumType>, [ArgSpec<UnsignedLongAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"uhksqrtus", RetValSpec<UnsignedShortAccumType>, [ArgSpec<UnsignedShortType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"uksqrtui", RetValSpec<UnsignedAccumType>, [ArgSpec<UnsignedIntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
      ]
  >;

  let Headers = [
    Assert,
    Sched,
    Stdfix,
    Strings,
  ];
}
