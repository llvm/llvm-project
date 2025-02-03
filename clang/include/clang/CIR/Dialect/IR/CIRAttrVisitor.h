#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H

#include "clang/CIR/Dialect/IR/CIRAttrs.h"

namespace cir {

template <typename ImplClass, typename RetTy> class CirAttrVisitor {
public:
  // FIXME: Create a TableGen list to automatically handle new attributes
  template <typename... Args>
  RetTy visit(mlir::Attribute attr, Args &&...args) {
    if (const auto intAttr = mlir::dyn_cast<cir::IntAttr>(attr))
      return static_cast<ImplClass *>(this)->visitCirIntAttr(
          intAttr, std::forward<Args>(args)...);
    if (const auto fltAttr = mlir::dyn_cast<cir::FPAttr>(attr))
      return static_cast<ImplClass *>(this)->visitCirFPAttr(
          fltAttr, std::forward<Args>(args)...);
    if (const auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
      return static_cast<ImplClass *>(this)->visitCirConstPtrAttr(
          ptrAttr, std::forward<Args>(args)...);
    llvm_unreachable("unhandled attribute type");
  }

  // If the implementation chooses not to implement a certain visit
  // method, fall back to the parent.
  template <typename... Args>
  RetTy visitCirIntAttr(cir::IntAttr attr, Args &&...args) {
    return static_cast<ImplClass *>(this)->visitCirAttr(
        attr, std::forward<Args>(args)...);
  }
  template <typename... Args>
  RetTy visitCirFPAttr(cir::FPAttr attr, Args &&...args) {
    return static_cast<ImplClass *>(this)->visitCirAttr(
        attr, std::forward<Args>(args)...);
  }
  template <typename... Args>
  RetTy visitCirConstPtrAttr(cir::ConstPtrAttr attr, Args &&...args) {
    return static_cast<ImplClass *>(this)->visitCirAttr(
        attr, std::forward<Args>(args)...);
  }

  template <typename... Args>
  RetTy visitCirAttr(mlir::Attribute attr, Args &&...args) {
    return RetTy();
  }
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H
