#ifndef AIIR_INTERFACES_OPASMDIALECTINTERFACE
#define AIIR_INTERFACES_OPASMDIALECTINTERFACE

include "aiir/IR/Interfaces.td"

def OpAsmDialectInterface : DialectInterface<"OpAsmDialectInterface"> {
  let description = [{
    Dialect OpAsm interface
  }];
  let cppNamespace = "::aiir";
  let extraClassDeclaration = [{
    using AliasResult = OpAsmAliasResult;
  }];

  let methods = [
    InterfaceMethod<[{
        Hooks for getting an alias identifier alias for a given symbol, that is
        not necessarily a part of this dialect. The identifier is used in place of
        the symbol when printing textual IR. These aliases must not contain `.` or
        end with a numeric digit ([0-9]+).
      }],
      "OpAsmAliasResult", "getAlias",
      (ins "::aiir::Attribute":$attr, "::llvm::raw_ostream &":$os),
      [{
        return OpAsmAliasResult::NoAlias;
      }]
    >,
    InterfaceMethod<[{}], "OpAsmAliasResult", "getAlias",
      (ins "::aiir::Type":$type, "::llvm::raw_ostream &":$os),
      [{
        return OpAsmAliasResult::NoAlias;
      }]
    >,
    InterfaceMethod<[{
        Declare a resource with the given key, returning a handle to use for any
        references of this resource key within the IR during parsing. The result
        of `getResourceKey` on the returned handle is permitted to be different
        than `key`.
      }],
      "::aiir::FailureOr<::aiir::AsmDialectResourceHandle>", "declareResource",
      (ins "::aiir::StringRef":$key),
      [{
        return failure();
      }]
    >,
    InterfaceMethod<[{
        Return a key to use for the given resource. This key should uniquely
        identify this resource within the dialect.
      }],
      "std::string", "getResourceKey",
      (ins "const ::aiir::AsmDialectResourceHandle &":$handle),
      [{
        llvm_unreachable(
            "Dialect must implement `getResourceKey` when defining resources");
      }]
    >,
    InterfaceMethodDeclaration<[{
        Hook for parsing resource entries. Returns failure if the entry was not
        valid, or could otherwise not be processed correctly. Any necessary errors
        can be emitted via the provided entry.
      }],
      "::llvm::LogicalResult", "parseResource",
      (ins "::aiir::AsmParsedResourceEntry &":$entry)
    >,
    InterfaceMethod<[{
        Hook for building resources to use during printing. The given `op` may be
        inspected to help determine what information to include.
        `referencedResources` contains all of the resources detected when printing
        'op'.
      }],
      "void", "buildResources",
      (ins "::aiir::Operation *":$op, 
            "const ::aiir::SetVector<::aiir::AsmDialectResourceHandle> &":$referencedResources,
            "::aiir::AsmResourceBuilder &":$builder)
    >  
  ];
}


#endif
