def GPUExtensions : StandardSpec<"GPUExtensions"> {
  HeaderSpec RPC = HeaderSpec<
    "gpu/rpc.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "rpc_reset",
            RetValSpec<VoidType>,
            [ArgSpec<UnsignedIntType>, ArgSpec<VoidPtr>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
