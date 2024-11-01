//===- FortranVariableInterface.td -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an interface for operations defining Fortran variables.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRANVARIABLEINTERFACE
#define FORTRANVARIABLEINTERFACE

include "mlir/IR/OpBase.td"


def fir_FortranVariableOpInterface : OpInterface<"FortranVariableOpInterface"> {
  let description = [{
    Interface for operations that create Fortran like variables in order to
    query about all their Fortran properties.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the address produced by the definition",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getBase",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getResult();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get Fortran attributes",
      /*retTy=*/"std::optional<fir::FortranVariableFlagsEnum>",
      /*methodName=*/"getFortranAttrs",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getFortran_attrs();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get the shape of the variable. May be a null value.",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getShape",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getShape();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Get explicit type parameters of the variable",
      /*retTy=*/"mlir::OperandRange",
      /*methodName=*/"getExplicitTypeParams",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = mlir::cast<ConcreteOp>(this->getOperation());
        return op.getTypeparams();
      }]
    >,
  ];

  let extraClassDeclaration = [{

    /// Get the sequence type or scalar value type corresponding to this
    /// variable.
    mlir::Type getElementOrSequenceType() {
      mlir::Type type = fir::unwrapPassByRefType(fir::unwrapRefType(getBase().getType()));
      if (auto boxCharType = type.dyn_cast<fir::BoxCharType>())
        return boxCharType.getEleTy();
      return type;
    }

    /// Get the scalar value type corresponding to this variable.
    mlir::Type getElementType() {
      return fir::unwrapSequenceType(getElementOrSequenceType());
    }

    /// Is the variable an array?
    bool isArray() {
      return getElementOrSequenceType().isa<fir::SequenceType>();
    }

    /// Return the rank of the entity if it is known at compile time.
    std::optional<unsigned> getRank() {
      if (auto sequenceType =
            getElementOrSequenceType().dyn_cast<fir::SequenceType>()) {
        if (sequenceType.hasUnknownShape())
          return {};
        return sequenceType.getDimension();
      }
      return 0;
    }

    /// Is this variable a Fortran pointer?
    bool isPointer() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::pointer);
    }

    /// Is this variable a Fortran allocatable?
    bool isAllocatable() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::allocatable);
    }

    /// Is this variable a Fortran optional?
    bool isOptional() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::optional);
    }

    /// Does this variable have the Fortran CONTIGUOUS attribute?
    /// Note that not having this attribute does not imply the
    /// variable is not contiguous.
    bool hasContiguousAttr() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::contiguous);
    }

    /// Is this a Fortran character variable?
    bool isCharacter() {
      return getElementType().isa<fir::CharacterType>();
    }

    /// Is this a Fortran character variable with an explicit length?
    bool hasExplicitCharLen() {
      return isCharacter() && !getExplicitTypeParams().empty();
    }

    /// Return the length of explicit length character variable.
    mlir::Value getExplicitCharLen() {
      assert(hasExplicitCharLen() && "must be an explicit length character");
      return getExplicitTypeParams()[0];
    }

    /// Is this variable represented as a fir.box or fir.class value?
    bool isBoxValue() {
      return getBase().getType().isa<fir::BaseBoxType>();
    }

    /// Is this variable represented as a fir.box or fir.class address?
    bool isBoxAddress() {
      return fir::isBoxAddress(getBase().getType());
    }

    /// Is this variable represented as the value or address of a fir.box or
    /// fir.class?
    bool isBox() {
      return fir::isBoxAddressOrValue(getBase().getType());
    }

    /// Is this variable a Fortran parameter?
    bool isParameter() {
      auto attrs = getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs,
                        fir::FortranVariableFlagsEnum::parameter);
    }

    /// Interface verifier imlementation for declare operations.
    mlir::LogicalResult verifyDeclareLikeOpImpl(mlir::Value memRef);

  }];

  let cppNamespace = "fir";

}

#endif  // FORTRANVARIABLEINTERFACE
