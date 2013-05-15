//===-- ExpressionSourceCode.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ExpressionSourceCode_h
#define liblldb_ExpressionSourceCode_h

#include "lldb/lldb-enumerations.h"

#include <string>

namespace lldb_private
{

class ExpressionSourceCode
{
public:
    static const char * g_expression_prefix;

    static ExpressionSourceCode *CreateWrapped (const char *prefix,
                                                const char *body)
    {
        return new ExpressionSourceCode ("$__lldb_expr",
                                         prefix,
                                         body,
                                         true);
    }
    
    static ExpressionSourceCode *CreateUnwrapped (const char *name,
                                                  const char *body)
    {
        return new ExpressionSourceCode (name,
                                         "",
                                         body,
                                         false);
    }
    
    bool NeedsWrapping () const
    {
        return m_wrap;
    }
    
    const char *GetName () const
    {
        return m_name.c_str();
    }
    
    bool GetText (std::string &text, 
                  lldb::LanguageType wrapping_language, 
                  bool const_object,
                  bool static_method) const;
    
private:
    ExpressionSourceCode (const char *name,
                          const char *prefix,
                          const char *body,
                          bool wrap) :
        m_name(name),
        m_prefix(prefix),
        m_body(body),
        m_wrap(wrap)
    {
    }
    
    std::string m_name;
    std::string m_prefix;
    std::string m_body;
    bool m_wrap;
};

} // namespace lldb_private

#endif
