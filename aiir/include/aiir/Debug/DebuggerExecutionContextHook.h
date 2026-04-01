//===- DebuggerExecutionContextHook.h - Debugger Support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of C API functions that are used by the debugger to
// interact with the ExecutionContext.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_SUPPORT_DEBUGGEREXECUTIONCONTEXTHOOK_H
#define AIIR_SUPPORT_DEBUGGEREXECUTIONCONTEXTHOOK_H

#include "aiir-c/IR.h"
#include "aiir/Debug/ExecutionContext.h"
#include "llvm/Support/Compiler.h"

extern "C" {
struct AIIRBreakpoint;
struct AIIRIRunit;
typedef struct AIIRBreakpoint *BreakpointHandle;
typedef struct AIIRIRunit *irunitHandle;

/// This is used by the debugger to control what to do after a breakpoint is
/// hit. See tracing::ExecutionContext::Control for more information.
void aiirDebuggerSetControl(int controlOption);

/// Print the available context for the current Action.
void aiirDebuggerPrintContext();

/// Print the current action backtrace.
void aiirDebuggerPrintActionBacktrace(bool withContext);

//===----------------------------------------------------------------------===//
// Cursor Management: The cursor is used to select an IRUnit from the context
// and to navigate through the IRUnit hierarchy.
//===----------------------------------------------------------------------===//

/// Print the current IR unit cursor.
void aiirDebuggerCursorPrint(bool withRegion);

/// Select the IR unit from the current context by ID.
void aiirDebuggerCursorSelectIRUnitFromContext(int index);

/// Select the parent IR unit of the provided IR unit, or print an error if the
/// IR unit has no parent.
void aiirDebuggerCursorSelectParentIRUnit();

/// Select the child IR unit at the provided index, print an error if the index
/// is out of bound. For example if the irunit is an operation, the children IR
/// units will be the operation's regions.
void aiirDebuggerCursorSelectChildIRUnit(int index);

/// Return the next IR unit logically in the IR. For example if the irunit is a
/// Region the next IR unit will be the next region in the parent operation or
/// nullptr if there is no next region.
void aiirDebuggerCursorSelectPreviousIRUnit();

/// Return the previous IR unit logically in the IR. For example if the irunit
/// is a Region, the previous IR unit will be the previous region in the parent
/// operation or nullptr if there is no previous region.
void aiirDebuggerCursorSelectNextIRUnit();

//===----------------------------------------------------------------------===//
// Breakpoint Management
//===----------------------------------------------------------------------===//

/// Enable the provided breakpoint.
void aiirDebuggerEnableBreakpoint(BreakpointHandle breakpoint);

/// Disable the provided breakpoint.
void aiirDebuggerDisableBreakpoint(BreakpointHandle breakpoint);

/// Add a breakpoint matching exactly the provided tag.
BreakpointHandle aiirDebuggerAddTagBreakpoint(const char *tag);

/// Add a breakpoint matching a pattern by name.
void aiirDebuggerAddRewritePatternBreakpoint(const char *patternNameInfo);

/// Add a breakpoint matching a file, line and column.
void aiirDebuggerAddFileLineColLocBreakpoint(const char *file, int line,
                                             int col);

} // extern "C"

namespace aiir {
// Setup the debugger hooks as a callback on the provided ExecutionContext.
void setupDebuggerExecutionContextHook(
    tracing::ExecutionContext &executionContext);

} // namespace aiir

#endif // AIIR_SUPPORT_DEBUGGEREXECUTIONCONTEXTHOOK_H
