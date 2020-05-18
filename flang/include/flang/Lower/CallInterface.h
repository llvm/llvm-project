//===-- Lower/CallInterface.h -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/IR/Function.h"
#include <memory>
#include <optional>
#include <string>

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::evaluate {
class ProcedureRef;
class ActualArgument;
namespace characteristics {
struct Procedure;
}
} // namespace Fortran::evaluate

namespace Fortran::lower {
class AbstractConverter;
class SymMap;
namespace pft {
struct FunctionLikeUnit;
}

/// PassedEntityTypes helps abstract whether CallInterface is mapping a
/// Symbol to mlir::Value (callee side) or an ActualArgument to a position
/// inside the input vector for the CallOp (caller side. It will be up to the
/// CallInterface user to produce the mlir::Value that will go in this input
/// vector).
class CallerInterface;
class CalleeInterface;
template <typename T>
struct PassedEntityTypes {};
template <>
struct PassedEntityTypes<CallerInterface> {
  using FortranEntity = const Fortran::evaluate::ActualArgument *;
  using FirValue = int;
};
template <>
struct PassedEntityTypes<CalleeInterface> {
  using FortranEntity = common::Reference<const semantics::Symbol>;
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
/// shared between both side (getting characteristics, mangled name, location).
/// It maps FIR arguments to front-end Symbol (callee side) or ActualArgument
/// (caller side) with the same code using the abstract FortranEntity type that
/// can be either a Symbol or an ActualArgument.
/// It works in two passes: a first pass over the characteristics that decides
/// how the interface must be. Then, the funcOp is created for it. Then a simple
/// pass over fir arguments finalize the interface information that must be
/// passed back to the user (and may require having the funcOp). All this
/// passes are driven from the CallInterface constructor.
template <typename T>
class CallInterface {
  friend CallInterfaceImpl<T>;

public:
  /// Enum the different ways an entity can be passed-by
  enum class PassEntityBy {
    BaseAddress,
    BoxChar,
    Box,
    AddressAndLength,
    /// Value means passed by value at the mlir level, it is not necessarily
    /// implied by Fortran Value attribute.
    Value
  };
  /// Different properties of an entity that can be passed/returned.
  /// One-to-One mapping with PassEntityBy but for
  /// PassEntityBy::AddressAndLength that has two properties.
  enum class Property {
    BaseAddress,
    BoxChar,
    CharAddress,
    CharLength,
    Box,
    Value
  };

  using FortranEntity = typename PassedEntityTypes<T>::FortranEntity;
  using FirValue = typename PassedEntityTypes<T>::FirValue;
  /// FirPlaceHolder are place holders for the mlir inputs and outputs that are
  /// created during the first pass before the mlir::FuncOp is created.
  struct FirPlaceHolder {
    /// Type for this input/output
    mlir::Type type;
    /// Position of related passedEntity in passedArguments.
    /// (passedEntity is the passedResult this value is resultEntityPosition.
    int passedEntityPosition;
    static constexpr int resultEntityPosition = -1;
    /// Indicate property of the entity passedEntityPosition that must be passed
    /// through this argument.
    Property property;
  };

  /// PassedEntity is what is provided back to the CallInterface user.
  /// It describe how the entity is plugged in the interface
  struct PassedEntity {
    /// How entity is passed by.
    PassEntityBy passBy;
    /// What is the entity (SymbolRef for callee/ActualArgument* for caller)
    /// What is the related mlir::FuncOp argument(s) (mlir::Value for callee /
    /// index for the caller).
    FortranEntity entity;
    FirValue firArgument;
    FirValue firLength; /* only for AddressAndLength */
  };

  /// Return the mlir::FuncOp. Note that front block is added by this
  /// utility if callee side.
  mlir::FuncOp getFuncOp() const { return func; }
  /// Number of MLIR inputs/outputs of the created FuncOp.
  std::size_t getNumFIRArguments() const { return inputs.size(); }
  std::size_t getNumFIRResults() const { return outputs.size(); }
  /// Return the MLIR output types.
  llvm::SmallVector<mlir::Type, 1> getResultType() const;

  /// Return a container of Symbol/ActualArgument* and how they must
  /// be plugged with the mlir::FuncOp.
  llvm::ArrayRef<PassedEntity> getPassedArguments() const {
    return passedArguments;
  }
  /// In case the result must be passed by the caller, indicate how.
  /// nullopt if the result is not passed by the caller.
  std::optional<PassedEntity> getPassedResult() const { return passedResult; }

private:
  /// CRTP handle.
  T &side() { return *static_cast<T *>(this); }
  /// buildImplicitInterface and buildExplicitInterface are the entry point
  /// of the first pass that define the interface and is required to get
  /// the mlir::FuncOp.
  void
  buildImplicitInterface(const Fortran::evaluate::characteristics::Procedure &);
  void
  buildExplicitInterface(const Fortran::evaluate::characteristics::Procedure &);
  /// Helper to get type after the first pass.
  mlir::FunctionType genFunctionType() const;
  /// Second pass entry point, once the mlir::FuncOp is created
  void mapBackInputToPassedEntity(const FirPlaceHolder &, FirValue);

  llvm::SmallVector<FirPlaceHolder, 1> outputs;
  llvm::SmallVector<FirPlaceHolder, 4> inputs;
  mlir::FuncOp func;
  llvm::SmallVector<PassedEntity, 4> passedArguments;
  std::optional<PassedEntity> passedResult;

protected:
  CallInterface(Fortran::lower::AbstractConverter &c) : converter{c} {}
  /// Entry point to be called by child ctor (childs need to be initialized
  /// first).
  void init();
  Fortran::lower::AbstractConverter &converter;
  /// Store characteristic once created, it is required for further information
  /// (e.g. getting the length of character result)
  std::unique_ptr<Fortran::evaluate::characteristics::Procedure> characteristic;
};

//===----------------------------------------------------------------------===//
// Caller side interface
//===----------------------------------------------------------------------===//

/// The CallerInterface provides the helpers needed by CallInterface
/// (getting the characteristic...) and a safe way for the user to
/// place the mlir::Value arguments into the input vector
/// once they are lowered.
class CallerInterface : public CallInterface<CallerInterface> {
public:
  CallerInterface(const Fortran::evaluate::ProcedureRef &p,
                  Fortran::lower::AbstractConverter &c)
      : CallInterface{c}, procRef{p} {
    init();
    actualInputs = llvm::SmallVector<mlir::Value, 3>(getNumFIRArguments());
  }
  /// CRTP callbacks
  bool hasAlternateReturns() const;
  std::string getMangledName() const;
  mlir::Location getCalleeLocation() const;
  Fortran::evaluate::characteristics::Procedure characterize() const;
  const Fortran::evaluate::ProcedureRef &getCallDescription() const {
    return procRef;
  };
  bool isMainProgram() const { return false; }

  /// Helpers to place the lowered arguments at the right place once they
  /// have been lowered.
  void placeInput(const PassedEntity &passedEntity, mlir::Value arg);
  void placeAddressAndLengthInput(const PassedEntity &passedEntity,
                                  mlir::Value addr, mlir::Value len);
  /// Get the input vector once it is complete.
  const llvm::SmallVector<mlir::Value, 3> &getInputs() const {
    assert(verifyActualInputs() && "lowered arguments are incomplete");
    return actualInputs;
  }
  /// Return result length when the function return non
  /// allocatable/pointer character.
  mlir::Value getResultLength();

private:
  /// Check that the input vector is complete.
  bool verifyActualInputs() const;
  const Fortran::evaluate::ProcedureRef &procRef;
  llvm::SmallVector<mlir::Value, 3> actualInputs;
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
    init();
  }
  bool hasAlternateReturns() const;
  std::string getMangledName() const;
  mlir::Location getCalleeLocation() const;
  Fortran::evaluate::characteristics::Procedure characterize() const;
  bool isMainProgram() const;
  Fortran::lower::pft::FunctionLikeUnit &getCallDescription() const {
    return funit;
  };

private:
  Fortran::lower::pft::FunctionLikeUnit &funit;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
