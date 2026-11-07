//===-- lib/Semantics/check-omp-device.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Semantic checks of OpenMP target device code, i.e. the bodies of TARGET
// constructs and procedures that are explicitly or implicitly DECLARE TARGET.

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_DEVICE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_DEVICE_H_

#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <vector>

namespace Fortran::semantics {

// Checks that no THREADPRIVATE variable is accessed in target device code.
//
// Accesses in the body of a TARGET construct and in the execution part of a
// procedure that is explicitly DECLARE TARGET for the device are diagnosed
// immediately. A procedure that is called from device code becomes implicitly
// DECLARE TARGET, which is only known once the whole compilation unit has been
// seen. Accesses and calls of all other procedures are therefore recorded, and
// those reachable from device code are diagnosed at the end of the compilation
// unit when compiling for a target device.
class OmpDeviceChecker : public virtual BaseChecker {
public:
  explicit OmpDeviceChecker(SemanticsContext &);

  void Leave(const parser::Program &);
  void Enter(const parser::SubroutineSubprogram &);
  void Leave(const parser::SubroutineSubprogram &);
  void Enter(const parser::FunctionSubprogram &);
  void Leave(const parser::FunctionSubprogram &);
  void Enter(const parser::SeparateModuleSubprogram &);
  void Leave(const parser::SeparateModuleSubprogram &);
  void Enter(const parser::ExecutionPart &);
  void Leave(const parser::ExecutionPart &);
  void Enter(const parser::EntryStmt &);

  void Leave(const parser::OmpBeginDirective &);
  void Leave(const parser::OmpBlockConstruct &);
  void Leave(const parser::OpenMPLoopConstruct &);
  void Enter(const parser::OmpClause &);
  void Leave(const parser::OmpClause &);
  void Enter(const parser::OmpObjectList &);
  void Leave(const parser::OmpObjectList &);
  void Enter(const parser::OmpArgumentList &);
  void Leave(const parser::OmpArgumentList &);
  void Enter(const parser::DataStmt &);
  void Leave(const parser::DataStmt &);

  void Enter(const parser::Name &);
  void Enter(const parser::Expr &);
  void Leave(const parser::Expr &);
  void Enter(const parser::CallStmt &);
  void Enter(const parser::AssignmentStmt &);

private:
  enum class DeviceContext { None, TargetRegion, DeclareTarget };

  // An access to a THREADPRIVATE variable, possibly through a statement
  // function.
  struct Access {
    parser::CharBlock source;
    const Symbol *variable;
    const Symbol *statementFunction;
  };

  // A call of a procedure. `caller` is the procedure containing the call
  // (null in a main program). `context` is the device context of the call.
  struct Call {
    parser::CharBlock source;
    const Symbol *callee;
    const Symbol *caller;
    DeviceContext context;
  };

  struct ProcedureInfo {
    // Whether the procedure is explicitly DECLARE TARGET, and whether that
    // makes it device code (i.e. DEVICE_TYPE is not HOST).
    bool isExplicitDeclareTarget{false};
    bool isExplicitDevice{false};
    std::vector<Access> accesses;
    std::vector<Call> calls;
  };

  void EnterProcedure(const Symbol *);
  void LeaveProcedure();
  void LeaveConstruct(const parser::OmpBeginDirective &);
  DeviceContext GetDeviceContext() const;
  const Symbol *CurrentProcedure() const;

  void CheckReference(parser::CharBlock, const Symbol &,
      const Symbol *statementFunction = nullptr);
  void CheckAccess(const Access &);
  void RecordCall(parser::CharBlock, const Symbol &);
  void RecordCalls(parser::CharBlock, const SomeExpr &);

  const Symbol *FindDefinition(const Symbol &) const;
  void CheckImplicitDeclareTarget();

  SemanticsContext &context_;
  // Whether implicit DECLARE TARGET procedures need to be checked.
  bool isTargetDevice_;
  // Procedures whose definitions are being walked (innermost last).
  std::vector<const Symbol *> procedures_;
  // Begin directives of the enclosing TARGET constructs whose body is being
  // walked.
  std::vector<const parser::OmpBeginDirective *> targets_;
  // Clauses being walked (innermost last).
  std::vector<llvm::omp::Clause> clauses_;
  // For each object list being walked, whether its objects are skipped.
  std::vector<bool> skippedObjectLists_;
  bool inExecutionPart_{false};
  // Number of enclosing parse tree nodes whose names are not checked.
  int skipDepth_{0};
  // Number of enclosing parser::Expr nodes.
  int exprDepth_{0};
  // Statement functions being expanded, to avoid infinite recursion.
  llvm::SmallPtrSet<const Symbol *, 4> expanding_;

  // All procedures defined in the compilation unit.
  llvm::DenseMap<const Symbol *, ProcedureInfo> procedureInfo_;
  // Symbols that refer to a procedure definition in the compilation unit:
  // ENTRY names and interfaces of separate module procedures.
  llvm::DenseMap<const Symbol *, const Symbol *> aliases_;
  // Calls from device code.
  std::vector<Call> deviceCalls_;
};

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_DEVICE_H_
