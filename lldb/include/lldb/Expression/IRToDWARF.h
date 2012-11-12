//===-- IRToDWARF.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IRToDWARF_h_
#define liblldb_IRToDWARF_h_

#include "llvm/Pass.h"
#include "llvm/PassManager.h"

#include "lldb/lldb-public.h"

class Relocator;
//----------------------------------------------------------------------
/// @class IRToDWARF IRToDWARF.h "lldb/Expression/IRToDWARF.h"
/// @brief Transforms the IR for a function into a DWARF location expression
///
/// Once an expression has been parsed and converted to IR, it can run
/// in two contexts: interpreted by LLDB as a DWARF location expression,
/// or compiled by the JIT and inserted into the target process for
/// execution.
///
/// IRToDWARF makes the first possible, by traversing the control flow
/// graph and writing the code for each basic block out as location
/// expression bytecode.  To ensure that the links between the basic blocks
/// remain intact, it uses a relocator that records the location of every
/// location expression instruction that has a relocatable operand, the
/// target of that operand (as a basic block), and the mapping of each basic
/// block to an actual location.  After all code has been written out, the
/// relocator post-processes it and performs all necessary relocations.
//----------------------------------------------------------------------
class IRToDWARF : public llvm::ModulePass
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] local_vars
    ///     A list of variables to populate with the local variables this
    ///     expression uses.
    ///
    /// @param[in] decl_map
    ///     The list of externally-referenced variables for the expression,
    ///     for use in looking up globals.
    ///
    /// @param[in] stream
    ///     The stream to dump DWARF bytecode onto.
    ///
    /// @param[in] func_name
    ///     The name of the function to translate to DWARF.
    //------------------------------------------------------------------
    IRToDWARF(lldb_private::ClangExpressionVariableList &local_vars, 
              lldb_private::ClangExpressionDeclMap *decl_map,
              lldb_private::StreamString &strm,
              const char* func_name = "$__lldb_expr");
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual ~IRToDWARF();
    
    //------------------------------------------------------------------
    /// Run this IR transformer on a single module
    ///
    /// @param[in] M
    ///     The module to run on.  This module is searched for the function
    ///     $__lldb_expr, and that function is converted to a location
    ///     expression.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool runOnModule(llvm::Module &M);
    
    //------------------------------------------------------------------
    /// Interface stub
    //------------------------------------------------------------------
    void assignPassManager(llvm::PMStack &PMS,
                           llvm::PassManagerType T = llvm::PMT_ModulePassManager);
    
    //------------------------------------------------------------------
    /// Returns PMT_ModulePassManager
    //------------------------------------------------------------------
    llvm::PassManagerType getPotentialPassManagerType() const;
private:
    //------------------------------------------------------------------
    /// Run this IR transformer on a single basic block
    ///
    /// @param[in] BB
    ///     The basic block to transform.
    ///
    /// @param[in] Relocator
    ///     The relocator to use when registering branches.
    ///
    /// @return
    ///     True on success; false otherwise
    //------------------------------------------------------------------
    bool runOnBasicBlock(llvm::BasicBlock &BB, Relocator &Relocator);
    
    std::string m_func_name;                                    ///< The name of the function to translate
    lldb_private::ClangExpressionVariableList &m_local_vars;   ///< The list of local variables to populate while transforming
    lldb_private::ClangExpressionDeclMap *m_decl_map;           ///< The list of external variables
    lldb_private::StreamString &m_strm;                         ///< The stream to write bytecode to
};

#endif
