//===- CIROpDefinitions.h - CIR op implementation helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the helpers shared by the CIROps<Partition>.cpp files:
// the custom directive parsers and printers referenced by the
// TableGen-generated op definitions, and the region terminator helpers. Each
// is implemented in the partition file of the op it belongs to.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_IR_CIROPDEFINITIONS_H
#define CLANG_LIB_CIR_DIALECT_IR_CIROPDEFINITIONS_H

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"

#include "mlir/IR/OpImplementation.h"

namespace cir {
class AssumeOp;
class EhDispatchOp;
class GlobalOp;
class IndirectBrOp;
class ScopeOp;
class SwitchFlatOp;
class TryOp;

// Checks that omitting the terminator of `region` is valid and, if so, creates
// and inserts the omitted `cir.yield` into the region. Implemented in
// CIROpsControlFlow.cpp.
mlir::LogicalResult ensureRegionTerm(mlir::OpAsmParser &parser,
                                     mlir::Region &region, llvm::SMLoc errLoc);

// True if the terminator of `r` should be omitted when printing.
bool omitRegionTerm(mlir::Region &r);

// Checks that the constant `attrType` is a valid initializer for `opType` on
// `op` (a cir.const or cir.global). Implemented in CIROpsCore.cpp.
mlir::LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                       mlir::Attribute attrType);

mlir::ParseResult parseOmittedTerminatorRegion(mlir::OpAsmParser &parser,
                                               mlir::Region &region);
void printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer, cir::ScopeOp &op,
                                  mlir::Region &region);

mlir::OptionalParseResult
parseGlobalMemorySpace(mlir::AsmParser &p,
                       mlir::ptr::MemorySpaceAttrInterface &attr);
void printGlobalMemorySpace(mlir::AsmPrinter &printer, cir::GlobalOp op,
                            mlir::ptr::MemorySpaceAttrInterface attr);

mlir::ParseResult parseAssumeBundle(
    mlir::OpAsmParser &p, cir::AssumeBundleKindAttr &bundleKindAttr,
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> &bundleArgs,
    llvm::SmallVector<mlir::Type, 1> &bundleArgTypes);
void printAssumeBundle(mlir::OpAsmPrinter &p, cir::AssumeOp op,
                       cir::AssumeBundleKindAttr kindAttr,
                       mlir::OperandRange bundleArgs,
                       mlir::TypeRange bundleArgTypes);

mlir::ParseResult parseIndirectBrOpSucessors(
    mlir::OpAsmParser &parser, mlir::Type &flagType,
    llvm::SmallVectorImpl<mlir::Block *> &succOperandBlocks,
    llvm::SmallVectorImpl<
        llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &succOperands,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Type>> &succOperandsTypes);
void printIndirectBrOpSucessors(mlir::OpAsmPrinter &p, cir::IndirectBrOp op,
                                mlir::Type flagType, mlir::SuccessorRange succs,
                                mlir::OperandRangeRange succOperands,
                                const mlir::TypeRangeRange &succOperandsTypes);

mlir::ParseResult parseSwitchFlatOpCases(
    mlir::OpAsmParser &parser, mlir::Type flagType, mlir::ArrayAttr &caseValues,
    llvm::SmallVectorImpl<mlir::Block *> &caseDestinations,
    llvm::SmallVectorImpl<
        llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &caseOperands,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Type>> &caseOperandTypes);
void printSwitchFlatOpCases(mlir::OpAsmPrinter &p, cir::SwitchFlatOp op,
                            mlir::Type flagType, mlir::ArrayAttr caseValues,
                            mlir::SuccessorRange caseDestinations,
                            mlir::OperandRangeRange caseOperands,
                            const mlir::TypeRangeRange &caseOperandTypes);

mlir::ParseResult parseComdatName(mlir::OpAsmParser &parser,
                                  mlir::StringAttr &comdatAttr);
void printComdatName(mlir::OpAsmPrinter &p, cir::GlobalOp op,
                     mlir::StringAttr comdat);

mlir::ParseResult parseGlobalOpTypeAndInitialValue(
    mlir::OpAsmParser &parser, mlir::TypeAttr &typeAttr,
    mlir::Attribute &initialValueAttr, mlir::Region &ctorRegion,
    mlir::Region &dtorRegion);
void printGlobalOpTypeAndInitialValue(mlir::OpAsmPrinter &p, cir::GlobalOp op,
                                      mlir::TypeAttr type,
                                      mlir::Attribute initAttr,
                                      mlir::Region &ctorRegion,
                                      mlir::Region &dtorRegion);

mlir::ParseResult parseCopyTypes(mlir::OpAsmParser &parser, mlir::Type &srcType,
                                 mlir::Type &dstType);
void printCopyTypes(mlir::OpAsmPrinter &printer, mlir::Operation *,
                    mlir::Type srcType, mlir::Type dstType);

mlir::ParseResult parseTryHandlerRegions(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<std::unique_ptr<mlir::Region>> &handlerRegions,
    mlir::ArrayAttr &handlerTypes);
void printTryHandlerRegions(mlir::OpAsmPrinter &printer, cir::TryOp op,
                            mlir::MutableArrayRef<mlir::Region> handlerRegions,
                            mlir::ArrayAttr handlerTypes);

mlir::ParseResult parseEhDispatchDestinations(
    mlir::OpAsmParser &parser, mlir::ArrayAttr &catchTypes,
    llvm::SmallVectorImpl<mlir::Block *> &catchDestinations,
    mlir::Block *&defaultDestination, mlir::UnitAttr &defaultIsCatchAll);
void printEhDispatchDestinations(mlir::OpAsmPrinter &p, cir::EhDispatchOp op,
                                 mlir::ArrayAttr catchTypes,
                                 mlir::SuccessorRange catchDestinations,
                                 mlir::Block *defaultDestination,
                                 mlir::UnitAttr defaultIsCatchAll);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_IR_CIROPDEFINITIONS_H
