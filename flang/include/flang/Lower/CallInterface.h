//===-- Lower/CallInterface.h -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
// Utility that defines fir call interface for procedure both on caller and
// and callee side and get the related FuncOp.
// It does not emit any FIR code but for the created mlir::FuncOp, instead it
// provides back a container of Symbol (callee side)/ActualArgument (caller
// side) with additional information for each element describing how it must be
// plugged with the mlir::FuncOp.
// It handles the fact that hidden arguments may be inserted for the result.
// while lowering.
//
// This utility uses the characteristic of Fortran procedures to operate, which
// is a term and concept used in Fortran to refer to the signature of a function
// or subroutine.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CALLINTERFACE_H
#define FORTRAN_LOWER_CALLINTERFACE_H

#include "flang/Common/reference.h"
#include "flang/Evaluate/characteristics.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <optional>

namespace Fortran::semantics {
class Symbol;
}

namespace mlir {
class Location;
}

namespace Fortran::lower {
class AbstractConverter;
namespace pft {
struct FunctionLikeUnit;
}

/// PassedEntityTypes helps abstract whether CallInterface is mapping a
/// Symbol to mlir::Value (callee side) or an ActualArgument to a position
/// inside the input vector for the CallOp (caller side. It will be up to the
/// CallInterface user to produce the mlir::Value that will go in this input
/// vector).
class CalleeInterface;
template <typename T>
struct PassedEntityTypes {};
template <>
struct PassedEntityTypes<CalleeInterface> {
  using FortranEntity =
      std::optional<common::Reference<const semantics::Symbol>>;
  using FirValue = mlir::Value;
};

/// Implementation helper
template <typename T>
class CallInterfaceImpl;

/// CallInterface defines all the logic to determine FIR function interfaces
/// from a characteristic, build the mlir::FuncOp and describe back the argument
/// mapping to its user.
/// The logic is shared between the callee and caller sides that it accepts as
/// a curiously recursive template to handle the few things that cannot be
/// shared between both sides (getting characteristics, mangled name, location).
/// It maps FIR arguments to front-end Symbol (callee side) or ActualArgument
/// (caller side) with the same code using the abstract FortranEntity type that
/// can be either a Symbol or an ActualArgument.
/// It works in two passes: a first pass over the characteristics that decides
/// how the interface must be. Then, the funcOp is created for it. Then a simple
/// pass over fir arguments finalizes the interface information that must be
/// passed back to the user (and may require having the funcOp). All these
/// passes are driven from the CallInterface constructor.
template <typename T>
class CallInterface {
  friend CallInterfaceImpl<T>;

public:
  /// Enum the different ways an entity can be passed-by
  enum class PassEntityBy {
    BaseAddress,
    BoxChar,
    // passing a read-only descriptor
    Box,
    // passing a writable descriptor
    MutableBox,
    AddressAndLength,
    /// Value means passed by value at the mlir level, it is not necessarily
    /// implied by Fortran Value attribute.
    Value,
    /// ValueAttribute means dummy has the the Fortran VALUE attribute.
    BaseAddressValueAttribute,
    CharBoxValueAttribute, // BoxChar with VALUE
    // Passing a character procedure as a <procedure address, result length>
    // tuple.
    CharProcTuple
  };

  /// Different properties of an entity that can be passed/returned.
  /// One-to-One mapping with PassEntityBy but for
  /// PassEntityBy::AddressAndLength that has two properties.
  enum class Property {
    BaseAddress,
    BoxChar,
    CharAddress,
    CharLength,
    CharProcTuple,
    Box,
    MutableBox,
    Value
  };

  using FortranEntity = typename PassedEntityTypes<T>::FortranEntity;
  using FirValue = typename PassedEntityTypes<T>::FirValue;

  /// FirPlaceHolder are place holders for the mlir inputs and outputs that are
  /// created during the first pass before the mlir::FuncOp is created.
  struct FirPlaceHolder {
    FirPlaceHolder(mlir::Type t, int passedPosition, Property p,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs)
        : type{t}, passedEntityPosition{passedPosition}, property{p},
          attributes{attrs.begin(), attrs.end()} {}
    /// Type for this input/output
    mlir::Type type;
    /// Position of related passedEntity in passedArguments.
    /// (passedEntity is the passedResult this value is resultEntityPosition).
    int passedEntityPosition;
    static constexpr int resultEntityPosition = -1;
    /// Indicate property of the entity passedEntityPosition that must be passed
    /// through this argument.
    Property property;
    /// MLIR attributes for this argument
    llvm::SmallVector<mlir::NamedAttribute> attributes;
  };

  /// PassedEntity is what is provided back to the CallInterface user.
  /// It describe how the entity is plugged in the interface
  struct PassedEntity {
    /// Is the dummy argument optional ?
    bool isOptional() const;
    /// Can the argument be modified by the callee ?
    bool mayBeModifiedByCall() const;
    /// Can the argument be read by the callee ?
    bool mayBeReadByCall() const;
    /// How entity is passed by.
    PassEntityBy passBy;
    /// What is the entity (SymbolRef for callee/ActualArgument* for caller)
    /// What is the related mlir::FuncOp argument(s) (mlir::Value for callee /
    /// index for the caller).
    FortranEntity entity;
    FirValue firArgument;
    FirValue firLength; /* only for AddressAndLength */

    /// Pointer to the argument characteristics. Nullptr for results.
    const Fortran::evaluate::characteristics::DummyArgument *characteristics =
        nullptr;
  };

  /// Return a container of Symbol/ActualArgument* and how they must
  /// be plugged with the mlir::FuncOp.
  llvm::ArrayRef<PassedEntity> getPassedArguments() const {
    return passedArguments;
  }
  /// In case the result must be passed by the caller, indicate how.
  /// nullopt if the result is not passed by the caller.
  std::optional<PassedEntity> getPassedResult() const { return passedResult; }
  /// Returns the mlir function type
  mlir::FunctionType genFunctionType();

  /// determineInterface is the entry point of the first pass that defines the
  /// interface and is required to get the mlir::FuncOp.
  void
  determineInterface(bool isImplicit,
                     const Fortran::evaluate::characteristics::Procedure &);

protected:
  CallInterface(Fortran::lower::AbstractConverter &c) : converter{c} {}
  /// CRTP handle.
  T &side() { return *static_cast<T *>(this); }
  /// Entry point to be called by child ctor to analyze the signature and
  /// create/find the mlir::FuncOp. Child needs to be initialized first.
  void declare();
  /// Second pass entry point, once the mlir::FuncOp is created.
  /// Nothing is done if it was already called.
  void mapPassedEntities();
  void mapBackInputToPassedEntity(const FirPlaceHolder &, FirValue);

  llvm::SmallVector<FirPlaceHolder> outputs;
  llvm::SmallVector<FirPlaceHolder> inputs;
  mlir::FuncOp func;
  llvm::SmallVector<PassedEntity> passedArguments;
  std::optional<PassedEntity> passedResult;

  Fortran::lower::AbstractConverter &converter;
  /// Store characteristic once created, it is required for further information
  /// (e.g. getting the length of character result)
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristic =
      std::nullopt;
};

//===----------------------------------------------------------------------===//
// Callee side interface
//===----------------------------------------------------------------------===//

/// CalleeInterface only provides the helpers needed by CallInterface
/// to abstract the specificities of the callee side.
class CalleeInterface : public CallInterface<CalleeInterface> {
public:
  CalleeInterface(Fortran::lower::pft::FunctionLikeUnit &f,
                  Fortran::lower::AbstractConverter &c)
      : CallInterface{c}, funit{f} {
    declare();
  }

  bool hasAlternateReturns() const;
  std::string getMangledName() const;
  mlir::Location getCalleeLocation() const;
  Fortran::evaluate::characteristics::Procedure characterize() const;
  bool isMainProgram() const;

  Fortran::lower::pft::FunctionLikeUnit &getCallDescription() const {
    return funit;
  }

  /// On the callee side it does not matter whether the procedure is
  /// called through pointers or not.
  bool isIndirectCall() const { return false; }

  /// Return the procedure symbol if this is a call to a user defined
  /// procedure.
  const Fortran::semantics::Symbol *getProcedureSymbol() const;

  /// Add mlir::FuncOp entry block and map fir block arguments to Fortran dummy
  /// argument symbols.
  mlir::FuncOp addEntryBlockAndMapArguments();

private:
  Fortran::lower::pft::FunctionLikeUnit &funit;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
