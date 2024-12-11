def GPUExtensions : StandardSpec<"GPUExtensions"> {
  HeaderSpec RPC = HeaderSpec<
    "gpu/rpc.h",
    [], // Macros
    [], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "rpc_host_call",
            RetValSpec<UnsignedLongLongType>,
            [ArgSpec<VoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
