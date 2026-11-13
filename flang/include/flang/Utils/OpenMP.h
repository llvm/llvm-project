//===-- include/flang/Utils/OpenMP.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_UTILS_OPENMP_H_
#define FORTRAN_UTILS_OPENMP_H_

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
class RecordType;
} // namespace fir

namespace Fortran::utils::openmp {
// TODO We can probably move the stuff inside `Support/OpenMP-utils.h/.cpp` here
// as well.

/// Create an `omp.map.info` op. See OpenMPOps.td for map operand semantics.
///
/// \param builder        - Operation builder.
/// \param loc            - Source location of the created op.
/// \param baseAddr       - Variable address. Box values are converted to their
/// data address with `fir.box_addr`.
/// \param varPtrPtr      - Optional address of the mapped pointer, used for
/// descriptor base-address maps.
/// \param name           - Name identifying the mapped object.
/// \param bounds         - Bounds selecting the mapped elements or section.
/// \param members        - Child map entries of a structured map.
/// \param membersIndex   - Placement indices of the child map entries.
/// \param mapType        - Map type and modifier flags.
/// \param mapCaptureType - Capture kind of the mapped variable.
/// \param retTy          - Result type. Replaced by the data address type when
/// \p baseAddr is a box value.
/// \param partialMap     - Whether the record is mapped only through selected
/// components rather than in its entirety.
/// \param mapperId       - Optional declare mapper symbol reference.
/// \param varPtrTy       - Pointer type used to derive the `var_ptr` type
/// attribute. Defaults to \p retTy after box conversion. Set this when an
/// opaque map result must retain the type of its FIR variable pointer.
mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    mlir::omp::ClauseMapFlags mapType,
    mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
    bool partialMap = false,
    mlir::FlatSymbolRefAttr mapperId = mlir::FlatSymbolRefAttr(),
    mlir::Type varPtrTy = mlir::Type());

/// For an mlir value that does not have storage, allocate temporary storage
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
mlir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp, mlir::Value val,
    llvm::StringRef name = "tmp.map");

/// Select how to repair values used in a target region but defined above it.
enum class RegionOutsiderHandling {
  /// Try to clone memory-effect-free producers used by the target entry block,
  /// and map remaining values through target map entries.
  CloneOrMap,
  /// Try to clone memory-effect-free producers used anywhere in the target
  /// region. Do not create new maps; fail if an outsider cannot be cloned.
  CloneOnly,
};

/// For values used inside a target region but defined outside, either clone
/// these values inside the target region or map them to the region.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - The target that needs to be extended by clones and/or
/// maps.
/// \param handling     - Specifies whether uncloneable outsiders can be mapped,
/// and which uses should be rewritten.
void cloneOrMapRegionOutsiders(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp,
    RegionOutsiderHandling handling = RegionOutsiderHandling::CloneOrMap);

using RecordMemberMapperMangler =
    std::function<void(std::string &mapperId, llvm::StringRef memberName)>;

/// Build the canonical symbol name for a derived type's default mapper from
/// the FIR record type. This matches the compiler-generated name shape used by
/// explicit default declare mapper lowering.
std::string getCanonicalDefaultDeclareMapperName(fir::RecordType recordType);

mlir::FlatSymbolRefAttr getOrGenImplicitDefaultDeclareMapper(
    fir::FirOpBuilder &firOpBuilder, mlir::Location loc,
    fir::RecordType recordType, llvm::StringRef mapperNameStr,
    RecordMemberMapperMangler mangler = {},
    bool emitAllocatableMemberMaps = true);
} // namespace Fortran::utils::openmp

#endif // FORTRAN_UTILS_OPENMP_H_
