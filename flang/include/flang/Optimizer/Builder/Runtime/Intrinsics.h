// Builder/Runtime/Intrinsics.h  Fortran runtime codegen interface -*- C++ -*-//
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
class Type;
class Value;
} // namespace aiir

namespace fir {
class CharBoxValue;
class FirOpBuilder;

namespace runtime {

aiir::Value genAssociated(fir::FirOpBuilder &, aiir::Location,
                          aiir::Value pointer, aiir::Value target);

void genPointerAssociate(fir::FirOpBuilder &, aiir::Location,
                         aiir::Value pointer, aiir::Value target);
void genPointerAssociateRemapping(fir::FirOpBuilder &, aiir::Location,
                                  aiir::Value pointer, aiir::Value target,
                                  aiir::Value bounds, bool isMonomorphic);

aiir::Value genCpuTime(fir::FirOpBuilder &, aiir::Location);
void genDateAndTime(fir::FirOpBuilder &, aiir::Location,
                    std::optional<fir::CharBoxValue> date,
                    std::optional<fir::CharBoxValue> time,
                    std::optional<fir::CharBoxValue> zone, aiir::Value values);

aiir::Value genDsecnds(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value refTime);

void genEtime(fir::FirOpBuilder &builder, aiir::Location loc,
              aiir::Value values, aiir::Value time);

void genFlush(fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value unit);

void genFree(fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value ptr);

aiir::Value genFseek(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value unit, aiir::Value offset, aiir::Value whence);
aiir::Value genFtell(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value unit);

aiir::Value genGetUID(fir::FirOpBuilder &, aiir::Location);
aiir::Value genGetGID(fir::FirOpBuilder &, aiir::Location);

aiir::Value genMalloc(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value size);

void genRandomInit(fir::FirOpBuilder &, aiir::Location, aiir::Value repeatable,
                   aiir::Value imageDistinct);
void genRandomNumber(fir::FirOpBuilder &, aiir::Location, aiir::Value harvest);
void genRandomSeed(fir::FirOpBuilder &, aiir::Location, aiir::Value size,
                   aiir::Value put, aiir::Value get);

/// generate rename runtime call
void genRename(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value path1, aiir::Value path2, aiir::Value status);

aiir::Value genSecnds(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value refTime);

/// generate time runtime call
aiir::Value genTime(fir::FirOpBuilder &builder, aiir::Location loc);

/// generate runtime call to transfer intrinsic with no size argument
void genTransfer(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Value resultBox, aiir::Value sourceBox,
                 aiir::Value moldBox);

/// generate runtime call to transfer intrinsic with size argument
void genTransferSize(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value resultBox, aiir::Value sourceBox,
                     aiir::Value moldBox, aiir::Value size);

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as aiir::Value{}
void genSystemClock(fir::FirOpBuilder &, aiir::Location, aiir::Value count,
                    aiir::Value rate, aiir::Value max);

// generate signal runtime call
// CALL SIGNAL(NUMBER, HANDLER [, STATUS])
// status can be {} or a value. It may also be dynamically absent
void genSignal(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value number, aiir::Value handler, aiir::Value status);

/// generate sleep runtime call
void genSleep(fir::FirOpBuilder &builder, aiir::Location loc,
              aiir::Value seconds);

/// generate chdir runtime call
aiir::Value genChdir(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value name);

aiir::Value genIrand(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value i);
aiir::Value genRand(fir::FirOpBuilder &builder, aiir::Location loc,
                    aiir::Value i);

/// generate dump of a descriptor
void genShowDescriptor(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value descriptor);

} // namespace runtime
} // namespace fir

#endif // FORTRAN_LOWER_RUNTIME_H
