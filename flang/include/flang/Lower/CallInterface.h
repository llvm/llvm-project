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
    FirPlaceHolder(mlir::Type t, int passedPosition, Property p)
        : type{t}, passedEntityPosition{passedPosition}, property{p} {}
    /// Type for this input/output
    mlir::Type type;
    /// Position of related passedEntity in passedArguments.
    /// (passedEntity is the passedResult this value is resultEntityPosition).
    int passedEntityPosition;
    static constexpr int resultEntityPosition = -1;
    /// Indicate property of the entity passedEntityPosition that must be passed
    /// through this argument.
    Property property;
  };

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

  llvm::SmallVector<FirPlaceHolder> outputs;
  mlir::FuncOp func;

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
