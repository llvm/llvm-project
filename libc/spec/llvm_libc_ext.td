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
              [ArgSpec<VoidPtr>,
               ArgSpec<SizeTType>]
          >,
      ]
  >;

  let Headers = [
    String,
  ];
}
