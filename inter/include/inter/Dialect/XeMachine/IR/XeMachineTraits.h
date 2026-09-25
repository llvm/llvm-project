#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINETRAITS_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINETRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::xemachine {

template <typename ConcreteType>
class NoMachineInst
    : public mlir::OpTrait::TraitBase<ConcreteType, NoMachineInst> {};

template <typename ConcreteType>
class NoAsmEmission
    : public mlir::OpTrait::TraitBase<ConcreteType, NoAsmEmission> {};

template <typename ConcreteType>
class CompletionFree
    : public mlir::OpTrait::TraitBase<ConcreteType, CompletionFree> {};

template <typename ConcreteType>
class FullScoreboardDrain
    : public mlir::OpTrait::TraitBase<ConcreteType, FullScoreboardDrain> {};

template <typename ConcreteType>
class Rematerializable
    : public mlir::OpTrait::TraitBase<ConcreteType, Rematerializable> {};

} // namespace mlir::OpTrait::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINETRAITS_H
