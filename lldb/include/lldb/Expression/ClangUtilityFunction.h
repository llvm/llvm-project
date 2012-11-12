//===-- ClangUtilityFunction.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangUtilityFunction_h_
#define liblldb_ClangUtilityFunction_h_

// C Includes
// C++ Includes
#include <string>
#include <map>
#include <memory>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ProcessDataAllocator.h"

namespace lldb_private 
{

//----------------------------------------------------------------------
/// @class ClangUtilityFunction ClangUtilityFunction.h "lldb/Expression/ClangUtilityFunction.h"
/// @brief Encapsulates a single expression for use with Clang
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  ClangUtilityFunction encapsulates
/// a self-contained function meant to be used from other code.  Utility
/// functions can perform error-checking for ClangUserExpressions, 
//----------------------------------------------------------------------
class ClangUtilityFunction : public ClangExpression
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] text
    ///     The text of the function.  Must be a full translation unit.
    ///
    /// @param[in] name
    ///     The name of the function, as used in the text.
    //------------------------------------------------------------------
    ClangUtilityFunction (const char *text, 
                          const char *name);
    
    virtual 
    ~ClangUtilityFunction ();

    //------------------------------------------------------------------
    /// Install the utility function into a process
    ///
    /// @param[in] error_stream
    ///     A stream to print parse errors and warnings to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to install the utility function to.
    ///
    /// @return
    ///     True on success (no errors); false otherwise.
    //------------------------------------------------------------------
    bool
    Install (Stream &error_stream, ExecutionContext &exe_ctx);
    
    //------------------------------------------------------------------
    /// Check whether the given PC is inside the function
    ///
    /// Especially useful if the function dereferences NULL to indicate a failed
    /// assert.
    ///
    /// @param[in] pc
    ///     The program counter to check.
    ///
    /// @return
    ///     True if the program counter falls within the function's bounds;
    ///     false if not (or the function is not JIT compiled)
    //------------------------------------------------------------------
    bool
    ContainsAddress (lldb::addr_t address)
    {
        // nothing is both >= LLDB_INVALID_ADDRESS and < LLDB_INVALID_ADDRESS,
        // so this always returns false if the function is not JIT compiled yet
        return (address >= m_jit_start_addr && address < m_jit_end_addr);
    }
    
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    const char *
    Text ()
    {
        return m_function_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    const char *
    FunctionName ()
    {
        return m_function_name.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    ClangExpressionDeclMap *
    DeclMap ()
    {
        return m_expr_decl_map.get();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when registering
    /// local variables.  May be NULL if the Expression doesn't care.
    //------------------------------------------------------------------
    ClangExpressionVariableList *
    LocalVariables ()
    {
        return NULL;
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should allow to access ASTs.
    /// May be NULL if the ASTs do not need to be transformed.
    ///
    /// @param[in] passthrough
    ///     The ASTConsumer that the returned transformer should send
    ///     the ASTs to after transformation.
    //------------------------------------------------------------------
    clang::ASTConsumer *
    ASTTransformer (clang::ASTConsumer *passthrough)
    {
        return NULL;
    }
    
    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    bool
    NeedsValidation ()
    {
        return false;
    }
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    bool
    NeedsVariableResolution ()
    {
        return false;
    }
    
private:
    std::auto_ptr<ClangExpressionDeclMap>   m_expr_decl_map;    ///< The map to use when parsing and materializing the expression.
    std::auto_ptr<ProcessDataAllocator>     m_data_allocator;   ///< The allocator for static data used in the expression.
    
    std::string                             m_function_text;    ///< The text of the function.  Must be a well-formed translation unit.
    std::string                             m_function_name;    ///< The name of the function.
};

} // namespace lldb_private

#endif  // liblldb_ClangUtilityFunction_h_
