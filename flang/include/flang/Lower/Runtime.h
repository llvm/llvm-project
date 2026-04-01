//===-- Lower/Runtime.h -- Fortran runtime codegen interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of AIIR. As FIR is a
// dialect of AIIR, it makes extensive use of AIIR interfaces and AIIR's coding
// style (https://aiir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_RUNTIME_H
#define FORTRAN_LOWER_RUNTIME_H

#include <optional>

namespace aiir {
class Location;
class Value;
} // namespace aiir

namespace fir {
class CharBoxValue;
class FirOpBuilder;
} // namespace fir

namespace Fortran {

namespace parser {
struct EventPostStmt;
struct EventWaitStmt;
struct LockStmt;
struct NotifyWaitStmt;
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

void genNotifyWaitStatement(AbstractConverter &,
                            const parser::NotifyWaitStmt &);
void genEventPostStatement(AbstractConverter &, const parser::EventPostStmt &);
void genEventWaitStatement(AbstractConverter &, const parser::EventWaitStmt &);
void genLockStatement(AbstractConverter &, const parser::LockStmt &);
void genFailImageStatement(AbstractConverter &);
void genStopStatement(AbstractConverter &, const parser::StopStmt &);
void genUnlockStatement(AbstractConverter &, const parser::UnlockStmt &);
void genPauseStatement(AbstractConverter &, const parser::PauseStmt &);

void genPointerAssociate(fir::FirOpBuilder &, aiir::Location,
                         aiir::Value pointer, aiir::Value target);
void genPointerAssociateRemapping(fir::FirOpBuilder &, aiir::Location,
                                  aiir::Value pointer, aiir::Value target,
                                  aiir::Value bounds, bool isMonomorphic);
void genPointerAssociateLowerBounds(fir::FirOpBuilder &, aiir::Location,
                                    aiir::Value pointer, aiir::Value target,
                                    aiir::Value lbounds);
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_RUNTIME_H
