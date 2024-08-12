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
          GuardedFunctionSpec<"daddf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"ddivf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"dfmaf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"dsqrtf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"dsubf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,


          GuardedFunctionSpec<"f16add", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16addf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16addl", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,

          GuardedFunctionSpec<"f16sub", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16subf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16subl", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          
          GuardedFunctionSpec<"faddf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"fdivf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"ffmaf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"fmulf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"dmulf128", RetValSpec<DoubleType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"f16mul", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16mulf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16mull", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,

          GuardedFunctionSpec<"f16div", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16divf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16divl", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,

          GuardedFunctionSpec<"f16fma", RetValSpec<Float16Type>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>, ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16fmaf", RetValSpec<Float16Type>, [ArgSpec<FloatType>, ArgSpec<FloatType>, ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16fmal", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,

          GuardedFunctionSpec<"f16sqrt", RetValSpec<Float16Type>, [ArgSpec<DoubleType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16sqrtf", RetValSpec<Float16Type>, [ArgSpec<FloatType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"f16sqrtl", RetValSpec<Float16Type>, [ArgSpec<LongDoubleType>], "LIBC_TYPES_HAS_FLOAT16">,

          GuardedFunctionSpec<"fsqrtf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"fsubf128", RetValSpec<FloatType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"powi", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>]>,
          FunctionSpec<"powif", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>]>,
      ]
  >;

  let Headers = [
    Assert,
    Math,
    Sched,
    Strings,
  ];
}
