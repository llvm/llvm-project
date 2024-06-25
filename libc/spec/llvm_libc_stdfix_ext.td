def LLVMLibcStdfixExt : StandardSpec<"llvm_libc_stdfix_ext"> {
  HeaderSpec StdFix = HeaderSpec<
      "stdfix.h",
      [],  // macros
      [],  // types
      [],  // enums
      [    // functions
          GuardedFunctionSpec<"exphk", RetValSpec<ShortAccumType>, [ArgSpec<ShortAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"expk", RetValSpec<AccumType>, [ArgSpec<AccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

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
    StdFix,
  ];
}
