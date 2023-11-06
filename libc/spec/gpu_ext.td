def RPCPortT : NamedType<"rpc_port_t">;
def RPCPortPtrT : PtrType<RPCPortT>;

def GPUExtensions : StandardSpec<"GPUExtensions"> {
  HeaderSpec RPC = HeaderSpec<
    "gpu/rpc.h",
    [], // Macros
    [RPCPortT, SizeTType], // Types
    [], // Enumerations
    [
        FunctionSpec<
            "rpc_host_call",
            RetValSpec<VoidType>,
            [ArgSpec<VoidPtr>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "rpc_open_port",
            RetValSpec<RPCPortT>,
            [ArgSpec<UnsignedIntType>]
        >,
        FunctionSpec<
            "rpc_send_n",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>, ArgSpec<VoidPtr>, ArgSpec<SizeTType>]
        >,
        FunctionSpec<
            "rpc_recv_n",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>, ArgSpec<VoidPtr>, ArgSpec<SizeTPtr>]
        >,
        FunctionSpec<
            "rpc_close_port",
            RetValSpec<VoidType>,
            [ArgSpec<RPCPortPtrT>]
        >,
    ]
  >;
  let Headers = [
    RPC,
  ];
}
