//===-- Lower/Runtime.h -- Fortran runtime codegen interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_RUNTIME_H
#define FORTRAN_LOWER_RUNTIME_H

namespace llvm {
template <typename T>
class Optional;
}

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace fir {
class CharBoxValue;
class FirOpBuilder;
} // namespace fir

namespace Fortran {

namespace parser {
struct EventPostStmt;
struct EventWaitStmt;
struct LockStmt;
struct PauseStmt;
struct StopStmt;
struct SyncAllStmt;
struct SyncImagesStmt;
struct SyncMemoryStmt;
struct SyncTeamStmt;
struct UnlockStmt;
} // namespace parser

namespace lower {

class AbstractConverter;

// Lowering of Fortran statement related runtime (other than IO and maths)

void genEventPostStatement(AbstractConverter &, const parser::EventPostStmt &);
void genEventWaitStatement(AbstractConverter &, const parser::EventWaitStmt &);
void genLockStatement(AbstractConverter &, const parser::LockStmt &);
void genFailImageStatement(AbstractConverter &);
void genStopStatement(AbstractConverter &, const parser::StopStmt &);
void genSyncAllStatement(AbstractConverter &, const parser::SyncAllStmt &);
void genSyncImagesStatement(AbstractConverter &,
                            const parser::SyncImagesStmt &);
void genSyncMemoryStatement(AbstractConverter &,
                            const parser::SyncMemoryStmt &);
void genSyncTeamStatement(AbstractConverter &, const parser::SyncTeamStmt &);
void genUnlockStatement(AbstractConverter &, const parser::UnlockStmt &);
void genPauseStatement(AbstractConverter &, const parser::PauseStmt &);

mlir::Value genAssociated(fir::FirOpBuilder &, mlir::Location,
                          mlir::Value pointer, mlir::Value target);

void genPointerAssociate(fir::FirOpBuilder &, mlir::Location,
                         mlir::Value pointer, mlir::Value target);
void genPointerAssociateRemapping(fir::FirOpBuilder &, mlir::Location,
                                  mlir::Value pointer, mlir::Value target,
                                  mlir::Value bounds);

mlir::Value genCpuTime(fir::FirOpBuilder &, mlir::Location);
void genDateAndTime(fir::FirOpBuilder &, mlir::Location,
                    llvm::Optional<fir::CharBoxValue> date,
                    llvm::Optional<fir::CharBoxValue> time,
                    llvm::Optional<fir::CharBoxValue> zone, mlir::Value values);

void genRandomInit(fir::FirOpBuilder &, mlir::Location, mlir::Value repeatable,
                   mlir::Value imageDistinct);
void genRandomNumber(fir::FirOpBuilder &, mlir::Location, mlir::Value harvest);
void genRandomSeed(fir::FirOpBuilder &, mlir::Location, mlir::Value size,
                   mlir::Value put, mlir::Value get);

/// generate runtime call to transfer intrinsic with no size argument
void genTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value sourceBox,
                 mlir::Value moldBox);

/// generate runtime call to transfer intrinsic with size argument
void genTransferSize(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value resultBox, mlir::Value sourceBox,
                     mlir::Value moldBox, mlir::Value size);

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as mlir::Value{}
void genSystemClock(fir::FirOpBuilder &, mlir::Location, mlir::Value count,
                    mlir::Value rate, mlir::Value max);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_RUNTIME_H
