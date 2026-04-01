//===-- TestInterfaces.td - Test dialect interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TEST_DIALECT_TEST_INTERFACES
#define AIIR_TEST_DIALECT_TEST_INTERFACES

include "aiir/IR/OpBase.td"
include "aiir/Interfaces/SideEffectInterfaceBase.td"

// A set of type interfaces used to test interface inheritance.
def TestBaseTypeInterfacePrintTypeA : TypeInterface<"TestBaseTypeInterfacePrintTypeA"> {
  let cppNamespace = "::test";
  let methods = [
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeA", (ins "::aiir::Location":$loc), [{
        emitRemark(loc) << $_type << " - TestA";
      }]
    >
  ];
}
def TestBaseTypeInterfacePrintTypeB
  : TypeInterface<"TestBaseTypeInterfacePrintTypeB", [TestBaseTypeInterfacePrintTypeA]> {
  let cppNamespace = "::test";
  let methods = [
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeB", (ins "::aiir::Location":$loc),
      [{}], /*defaultImplementation=*/[{
        emitRemark(loc) << $_type << " - TestB";
      }]
    >
  ];
}

// A type interface used to test the ODS generation of type interfaces.
def TestTypeInterface
  : TypeInterface<"TestTypeInterface", [TestBaseTypeInterfacePrintTypeB]> {
  let cppNamespace = "::test";
  let methods = [
    InterfaceMethod<"Prints the type name.",
      "void", "printTypeC", (ins "::aiir::Location":$loc)
    >,
    // Check that we can have multiple method with the same name.
    InterfaceMethod<"Prints the type name, with a value prefixed.",
      "void", "printTypeC", (ins "::aiir::Location":$loc, "int":$value)
    >,
    InterfaceMethod<"Prints the type name, with a value prefixed.",
      "void", "printTypeC", (ins "::aiir::Location":$loc, "float":$value),
      [{}], /*defaultImplementation=*/[{
        emitRemark(loc) << $_type << " - " << value << " - Float TestC";
      }]
    >,
    // It should be possible to use the interface type name as result type
    // as well as in the implementation.
    InterfaceMethod<"Prints the type name and returns the type as interface.",
      "TestTypeInterface", "printTypeRet", (ins "::aiir::Location":$loc),
      [{}], /*defaultImplementation=*/[{
        emitRemark(loc) << $_type << " - TestRet";
        return $_type;
      }]
    >,
  ];
  let extraClassDeclaration = [{
    /// Prints the type name.
    void printTypeD(::aiir::Location loc) const {
      emitRemark(loc) << *this << " - TestD";
    }
  }];
  let extraTraitClassDeclaration = [{
    /// Prints the type name.
    void printTypeE(::aiir::Location loc) const {
      emitRemark(loc) << $_type << " - TestE";
    }
  }];
}

def TestExternalTypeInterface : TypeInterface<"TestExternalTypeInterface"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"Returns the bitwidth of the type plus 'arg'.",
      "unsigned", "getBitwidthPlusArg", (ins "unsigned":$arg)>,
    StaticInterfaceMethod<"Returns some value plus 'arg'.",
      "unsigned", "staticGetSomeValuePlusArg", (ins "unsigned":$arg)>,
    InterfaceMethod<"Returns the argument doubled.",
      "unsigned", "getBitwidthPlusDoubleArgument", (ins "unsigned":$arg), "",
      "return $_type.getIntOrFloatBitWidth() + 2 * arg;">,
    StaticInterfaceMethod<"Returns the argument.",
      "unsigned", "staticGetArgument", (ins "unsigned":$arg), "",
      "return arg;">,
  ];
}

def TestExternalFallbackTypeInterface
    : TypeInterface<"TestExternalFallbackTypeInterface"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"Returns the bitwidth of the given integer type.",
      "unsigned", "getBitwidth", (ins), "", "return $_type.getWidth();">,
  ];
}

def TestExternalAttrInterface : AttrInterface<"TestExternalAttrInterface"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"Gets the dialect pointer.", "const ::aiir::Dialect *",
      "getDialectPtr">,
    StaticInterfaceMethod<"Returns some number.", "int", "getSomeNumber">,
  ];
}

def TestExternalOpInterface : OpInterface<"TestExternalOpInterface"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"Returns the length of the operation name plus arg.",
      "unsigned", "getNameLengthPlusArg", (ins "unsigned":$arg)>,
    StaticInterfaceMethod<
      "Returns the length of the operation name plus arg twice.", "unsigned",
      "getNameLengthPlusArgTwice", (ins "unsigned":$arg)>,
    InterfaceMethod<
      "Returns the length of the product of the operation name and arg.",
      "unsigned", "getNameLengthTimesArg", (ins "unsigned":$arg), "",
      "return arg * $_op->getName().getStringRef().size();">,
    StaticInterfaceMethod<"Returns the length of the operation name minus arg.",
      "unsigned", "getNameLengthMinusArg", (ins "unsigned":$arg), "",
      "return ConcreteOp::getOperationName().size() - arg;">,
  ];
}

def TestEffectOpInterface
    : EffectOpInterfaceBase<"TestEffectOpInterface",
                            "::aiir::TestEffects::Effect"> {
  let cppNamespace = "::aiir";
}

class TestEffect<string effectName>
    : SideEffect<TestEffectOpInterface, effectName, DefaultResource, 0,
                 PartialEffect>;

class TestEffects<list<TestEffect> effects = []>
   : SideEffectsTraitBase<TestEffectOpInterface, effects>;

def TestConcreteEffect : TestEffect<"TestEffects::Concrete">;

def TestOptionallyImplementedOpInterface
    : OpInterface<"TestOptionallyImplementedOpInterface"> {
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"", "bool", "getImplementsInterface", (ins)>,
  ];

  let extraClassOf = [{
    return $_op.getImplementsInterface();
  }];
}

def TestOptionallyImplementedAttrInterface
    : AttrInterface<"TestOptionallyImplementedAttrInterface"> {
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"", "bool", "getImplementsInterface", (ins)>,
  ];

  let extraClassOf = [{
    return $_attr.getImplementsInterface();
  }];
}

def TestOptionallyImplementedTypeInterface
    : TypeInterface<"TestOptionallyImplementedTypeInterface"> {
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"", "bool", "getImplementsInterface", (ins)>,
  ];

  let extraClassOf = [{
    return $_type.getImplementsInterface();
  }];
}

// Dummy type interface "A" that requires type interface "B" to be complete.
def TestCyclicTypeInterfaceA : TypeInterface<"TestCyclicTypeInterfaceA"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"",
      "::aiir::FailureOr<::aiir::TestCyclicTypeInterfaceB>",
      /*methodName=*/"returnB",
      (ins),
      /*methodBody=*/"",
      /*defaultImpl=*/"return aiir::failure();"
    >,
  ];
}

// Dummy type interface "B" that requires type interface "A" to be complete.
def TestCyclicTypeInterfaceB : TypeInterface<"TestCyclicTypeInterfaceB"> {
  let cppNamespace = "::aiir";
  let methods = [
    InterfaceMethod<"",
      "::aiir::FailureOr<::aiir::TestCyclicTypeInterfaceA>",
      /*methodName=*/"returnA",
      (ins),
      /*methodBody=*/"",
      /*defaultImpl=*/"return aiir::failure();"
    >,
  ];
}

#endif // AIIR_TEST_DIALECT_TEST_INTERFACES
