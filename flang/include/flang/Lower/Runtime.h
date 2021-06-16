//===-- Lower/Runtime.h -- Fortran runtime codegen interface ----*- C++ -*-===//
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
}

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
class FirOpBuilder;

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

void genDateAndTime(FirOpBuilder &, mlir::Location,
                    llvm::Optional<fir::CharBoxValue> date,
                    llvm::Optional<fir::CharBoxValue> time,
                    llvm::Optional<fir::CharBoxValue> zone);

void genRandomInit(FirOpBuilder &, mlir::Location, mlir::Value repeatable,
                   mlir::Value imageDistinct);
void genRandomNumber(FirOpBuilder &, mlir::Location, mlir::Value harvest);
void genRandomSeed(FirOpBuilder &, mlir::Location, int argIndex,
                   mlir::Value argBox);

/// generate runtime call to transfer intrinsic with no size argument
void genTransfer(Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value sourceBox,
                 mlir::Value moldBox);

/// generate runtime call to transfer intrinsic with size argument
void genTransferSize(Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value resultBox, mlir::Value sourceBox,
                     mlir::Value moldBox, mlir::Value size);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_RUNTIME_H
