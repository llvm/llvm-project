//===-- include/flang/Utils/OpenMP.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_UTILS_OPENMP_H_
#define FORTRAN_UTILS_OPENMP_H_

#include "aiir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
class RecordType;
} // namespace fir

namespace Fortran::utils::openmp {
// TODO We can probably move the stuff inside `Support/OpenMP-utils.h/.cpp` here
// as well.

/// Create an `omp.map.info` op. Parameters other than the ones documented below
/// correspond to operation arguments in the OpenMPOps.td file, see op docs for
/// more details.
///
/// \param [in] builder - AIIR operation builder.
/// \param [in] loc     - Source location of the created op.
aiir::omp::MapInfoOp createMapInfoOp(aiir::OpBuilder &builder,
    aiir::Location loc, aiir::Value baseAddr, aiir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<aiir::Value> bounds,
    llvm::ArrayRef<aiir::Value> members, aiir::ArrayAttr membersIndex,
    aiir::omp::ClauseMapFlags mapType,
    aiir::omp::VariableCaptureKind mapCaptureType, aiir::Type retTy,
    bool partialMap = false,
    aiir::FlatSymbolRefAttr mapperId = aiir::FlatSymbolRefAttr());

/// For an aiir value that does not have storage, allocate temporary storage
/// (outside the target region), store the value in that storage, and map the
/// storage to the target region.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - Target op to which the temporary value is mapped.
/// \param val          - Temp value that should be mapped to the target region.
/// \param name         - A string used to identify the created `omp.map.info`
/// op.
///
/// \returns The loaded mapped value inside the target region.
aiir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    aiir::omp::TargetOp targetOp, aiir::Value val,
    llvm::StringRef name = "tmp.map");

/// For values used inside a target region but defined outside, either clone
/// these value inside the target region or map them to the region. This
/// function first tries to clone values (if they are defined by
/// memory-effect-free ops, otherwise, the values are mapped.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - The target that needs to be extended by clones and/or
/// maps.
void cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, aiir::omp::TargetOp targetOp);

using RecordMemberMapperMangler =
    std::function<void(std::string &mapperId, llvm::StringRef memberName)>;

aiir::FlatSymbolRefAttr getOrGenImplicitDefaultDeclareMapper(
    fir::FirOpBuilder &firOpBuilder, aiir::Location loc,
    fir::RecordType recordType, llvm::StringRef mapperNameStr,
    RecordMemberMapperMangler mangler = {});
} // namespace Fortran::utils::openmp

#endif // FORTRAN_UTILS_OPENMP_H_
