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
        FunctionSpec<
            "rpc_host_call",
            RetValSpec<VoidType>,
            [ArgSpec<VoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
