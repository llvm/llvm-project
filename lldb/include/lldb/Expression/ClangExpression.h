//===-- ClangExpression.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpression_h_
#define liblldb_ClangExpression_h_

// C Includes
// C++ Includes
#include <string>
#include <map>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Target/Process.h"

namespace lldb_private {

class RecordingMemoryManager;

//----------------------------------------------------------------------
/// @class ClangExpression ClangExpression.h "lldb/Expression/ClangExpression.h"
/// @brief Encapsulates a single expression for use with Clang
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  ClangExpression encapsulates
/// the objects needed to parse and interpret or JIT an expression.  It
/// uses the Clang parser to produce LLVM IR from the expression.
//----------------------------------------------------------------------
class ClangExpression
{
public:
    enum ResultType {
        eResultTypeAny,
        eResultTypeId
    };
    
    ClangExpression () :
        m_jit_process_wp(),
        m_jit_start_addr (LLDB_INVALID_ADDRESS),
        m_jit_end_addr (LLDB_INVALID_ADDRESS)
    {
    }

    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual ~ClangExpression ()
    {
    }
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    virtual const char *
    Text () = 0;
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    virtual const char *
    FunctionName () = 0;
    
    //------------------------------------------------------------------
    /// Return the language that should be used when parsing.  To use
    /// the default, return eLanguageTypeUnknown.
    //------------------------------------------------------------------
    virtual lldb::LanguageType
    Language ()
    {
        return lldb::eLanguageTypeUnknown;
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    virtual ClangExpressionDeclMap *
    DeclMap () = 0;
    
    //------------------------------------------------------------------
    /// Return the object that the parser should allow to access ASTs.
    /// May be NULL if the ASTs do not need to be transformed.
    ///
    /// @param[in] passthrough
    ///     The ASTConsumer that the returned transformer should send
    ///     the ASTs to after transformation.
    //------------------------------------------------------------------
    virtual clang::ASTConsumer *
    ASTTransformer (clang::ASTConsumer *passthrough) = 0;
    
    //------------------------------------------------------------------
    /// Return the desired result type of the function, or 
    /// eResultTypeAny if indifferent.
    //------------------------------------------------------------------
    virtual ResultType
    DesiredResultType ()
    {
        return eResultTypeAny;
    }
    
    //------------------------------------------------------------------
    /// Flags
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    virtual bool
    NeedsValidation () = 0;
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    virtual bool
    NeedsVariableResolution () = 0;

    //------------------------------------------------------------------
    /// Return the address of the function's JIT-compiled code, or
    /// LLDB_INVALID_ADDRESS if the function is not JIT compiled
    //------------------------------------------------------------------
    lldb::addr_t
    StartAddress ()
    {
        return m_jit_start_addr;
    }

protected:

    lldb::ProcessWP m_jit_process_wp;
    lldb::addr_t    m_jit_start_addr;       ///< The address of the JITted function within the JIT allocation.  LLDB_INVALID_ADDRESS if invalid.
    lldb::addr_t    m_jit_end_addr;         ///< The address of the JITted function within the JIT allocation.  LLDB_INVALID_ADDRESS if invalid.

};

} // namespace lldb_private

#endif  // liblldb_ClangExpression_h_
