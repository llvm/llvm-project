def GPUExtensions : StandardSpec<"GPUExtensions"> {
  HeaderSpec RPC = HeaderSpec<
    "gpu/rpc.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "rpc_host_call",
            RetValSpec<VoidType>,
            [ArgSpec<VoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "rpc_fprintf",
            RetValSpec<IntType>,
            [ArgSpec<FILERestrictedPtr>,
             ArgSpec<ConstCharRestrictedPtr>,
             ArgSpec<VoidPtr>,
             ArgSpec<SizeTType>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
