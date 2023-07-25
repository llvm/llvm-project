//==-- Builder/PPCIntrinsicCall.h - lowering of PowerPC intrinsics -*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_PPCINTRINSICCALL_H
#define FORTRAN_LOWER_PPCINTRINSICCALL_H

#include "flang/Common/static-multimap-view.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace fir {

/// Enums used to templatize vector intrinsic function generators. Enum does
/// not contain every vector intrinsic, only intrinsics that share generators.
enum class VecOp {
  Abs,
  Add,
  And,
  Anyge,
  Cmpge,
  Cmpgt,
  Cmple,
  Cmplt,
  Convert,
  Ctf,
  Cvf,
  Nmadd,
  Msub,
  Mul,
  Sel,
  Sl,
  Sld,
  Sldw,
  Sll,
  Slo,
  Sr,
  Srl,
  Sro,
  Sub,
  Xor
};

// Wrapper struct to encapsulate information for a vector type. Preserves
// sign of eleTy if eleTy is signed/unsigned integer. Helps with vector type
// conversions.
struct VecTypeInfo {
  mlir::Type eleTy;
  uint64_t len;

  mlir::Type toFirVectorType() { return fir::VectorType::get(len, eleTy); }

  // We need a builder to do the signless element conversion.
  mlir::Type toMlirVectorType(mlir::MLIRContext *context) {
    // Will convert to eleTy to signless int if eleTy is signed/unsigned int.
    auto convEleTy{getConvertedElementType(context, eleTy)};
    return mlir::VectorType::get(len, convEleTy);
  }

  bool isFloat32() { return mlir::isa<mlir::Float32Type>(eleTy); }

  bool isFloat64() { return mlir::isa<mlir::Float64Type>(eleTy); }

  bool isFloat() { return isFloat32() || isFloat64(); }
};

//===----------------------------------------------------------------------===//
// Helper functions for argument handling in vector intrinsics.
//===----------------------------------------------------------------------===//

// Returns a VecTypeInfo with element type and length of given fir vector type.
// Preserves signness of fir vector type if element type of integer.
static inline VecTypeInfo getVecTypeFromFirType(mlir::Type firTy) {
  assert(firTy.isa<fir::VectorType>());
  VecTypeInfo vecTyInfo;
  vecTyInfo.eleTy = firTy.dyn_cast<fir::VectorType>().getEleTy();
  vecTyInfo.len = firTy.dyn_cast<fir::VectorType>().getLen();
  return vecTyInfo;
}

static inline VecTypeInfo getVecTypeFromFir(mlir::Value firVec) {
  return getVecTypeFromFirType(firVec.getType());
}

// Converts array of fir vectors to mlir vectors.
static inline llvm::SmallVector<mlir::Value, 4>
convertVecArgs(fir::FirOpBuilder &builder, mlir::Location loc,
               VecTypeInfo vecTyInfo, llvm::SmallVector<mlir::Value, 4> args) {
  llvm::SmallVector<mlir::Value, 4> newArgs;
  auto ty{vecTyInfo.toMlirVectorType(builder.getContext())};
  assert(ty && "unknown mlir vector type");
  for (size_t i = 0; i < args.size(); i++)
    newArgs.push_back(builder.createConvert(loc, ty, args[i]));
  return newArgs;
}

// This overload method is used only if arguments are of different types.
static inline llvm::SmallVector<mlir::Value, 4>
convertVecArgs(fir::FirOpBuilder &builder, mlir::Location loc,
               llvm::SmallVectorImpl<VecTypeInfo> &vecTyInfo,
               llvm::SmallVector<mlir::Value, 4> args) {
  llvm::SmallVector<mlir::Value, 4> newArgs;
  for (size_t i = 0; i < args.size(); i++) {
    mlir::Type ty{vecTyInfo[i].toMlirVectorType(builder.getContext())};
    assert(ty && "unknown mlir vector type");
    newArgs.push_back(builder.createConvert(loc, ty, args[i]));
  }
  return newArgs;
}

struct PPCIntrinsicLibrary : IntrinsicLibrary {

  // Constructors.
  explicit PPCIntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : IntrinsicLibrary(builder, loc) {}
  PPCIntrinsicLibrary() = delete;
  PPCIntrinsicLibrary(const PPCIntrinsicLibrary &) = delete;

  // PPC intrinsic handlers.
  template <bool isImm>
  void genMtfsf(llvm::ArrayRef<fir::ExtendedValue>);

  fir::ExtendedValue genVecAbs(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue
  genVecAddAndMulSubXor(mlir::Type resultType,
                        llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecCmp(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecConvert(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecAnyCompare(mlir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecNmaddMsub(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecShift(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);

  fir::ExtendedValue genVecSel(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);
};

const IntrinsicHandler *findPPCIntrinsicHandler(llvm::StringRef name);

std::pair<const MathOperation *, const MathOperation *>
checkPPCMathOperationsRange(llvm::StringRef name);

} // namespace fir

#endif // FORTRAN_LOWER_PPCINTRINSICCALL_H
