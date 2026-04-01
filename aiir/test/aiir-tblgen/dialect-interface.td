// RUN: aiir-tblgen -gen-dialect-interface-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL

include "aiir/IR/Interfaces.td"

def NoDefaultMethod : DialectInterface<"NoDefaultMethod"> {
  let description = [{
    This is an example dialect interface without default method body.
  }];

  let cppNamespace = "::aiir::example";

  let methods = [
    InterfaceMethod<
      "Check if it's an example dialect", "bool", "isExampleDialect", (ins)
    >,
    InterfaceMethod<
      "second method to check if multiple methods supported",
      "unsigned", "supportSecondMethod", (ins "::aiir::Type":$type)
      >
  ];
}

// DECL:   class NoDefaultMethod : public {{.*}}DialectInterface::Base<NoDefaultMethod>
// DECL:   public:
// DECL:   virtual bool isExampleDialect() const {}
// DECL:   virtual unsigned supportSecondMethod(::aiir::Type type) const {}
// DECL:   NoDefaultMethod(::aiir::Dialect *dialect) : Base(dialect) {}

def WithDefaultMethodInterface : DialectInterface<"WithDefaultMethodInterface"> {
  let description = [{
    This is an example dialect interface with default method bodies.
  }];

  let cppNamespace = "::aiir::example";

  let methods = [
      InterfaceMethod<
        "Check if it's an example dialect", "bool", "isExampleDialect", (ins),
        [{
          return true;
        }]
      >,
      InterfaceMethod<
        "second method to check if multiple methods supported",
        "unsigned", "supportSecondMethod", (ins "::aiir::Type":$type)
      >
  ];
}

// DECL:   virtual bool isExampleDialect() const {
// DECL-NEXT:   return true;
// DECL-NEXT:   }

def OnlyDeclarationInterfaceWithExtraDecls : DialectInterface<"OnlyDeclarationInterfaceWithExtraDecls"> {
  let description = [{
    This is an example dialect interface with only declarations.
  }];

  let cppNamespace = "::aiir::example";

  let methods = [
      InterfaceMethodDeclaration<
        "a method declaration", "void", "exampleMethodDeclaration", 
        (ins "::aiir::Type":$type)
      >
  ];

  let extraClassDeclaration = [{
    using DeclType = int;
  }];
}

// DECL:   class OnlyDeclarationInterfaceWithExtraDecls : public {{.*}}DialectInterface::Base<OnlyDeclarationInterfaceWithExtraDecls>
// DECL:   virtual void exampleMethodDeclaration(::aiir::Type type) const;
// DECL:   using DeclType = int;

def PureVirtualInterface : DialectInterface<"PureVirtualInterface"> {
  let description = [{
    This is an example dialect interface with pure virtual methods.
  }];

  let cppNamespace = "::aiir::example";

  let methods = [
      PureVirtualInterfaceMethod<
        "Check if it's an example dialect", "bool", "isExampleDialect",
        (ins)
      >
  ];
}

// DECL:   class PureVirtualInterface : public {{.*}}DialectInterface::Base<PureVirtualInterface>
// DECL:   public:
// DECL:   virtual bool isExampleDialect() const = 0;
// DECL:   protected:
// DECL-NEXT:   PureVirtualInterface(::aiir::Dialect *dialect) : Base(dialect) {}
