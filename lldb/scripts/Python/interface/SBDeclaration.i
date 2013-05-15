//===-- SWIG Interface for SBDeclaration --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
    %feature("docstring",
    "Specifies an association with a line and column for a variable."
    ) SBDeclaration;
    class SBDeclaration
    {
        public:
        
        SBDeclaration ();
        
        SBDeclaration (const lldb::SBDeclaration &rhs);
        
        ~SBDeclaration ();
        
        bool
        IsValid () const;
        
        lldb::SBFileSpec
        GetFileSpec () const;
        
        uint32_t
        GetLine () const;
        
        uint32_t
        GetColumn () const;
        
        bool
        GetDescription (lldb::SBStream &description);
        
        void
        SetFileSpec (lldb::SBFileSpec filespec);
        
        void
        SetLine (uint32_t line);
        
        void
        SetColumn (uint32_t column);
        
        bool
        operator == (const lldb::SBDeclaration &rhs) const;
        
        bool
        operator != (const lldb::SBDeclaration &rhs) const;
        
        %pythoncode %{
            __swig_getmethods__["file"] = GetFileSpec
            if _newclass: file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this line entry.''')
            
            __swig_getmethods__["line"] = GetLine
            if _newclass: ling = property(GetLine, None, doc='''A read only property that returns the 1 based line number for this line entry, a return value of zero indicates that no line information is available.''')
            
            __swig_getmethods__["column"] = GetColumn
            if _newclass: column = property(GetColumn, None, doc='''A read only property that returns the 1 based column number for this line entry, a return value of zero indicates that no column information is available.''')
            %}
        
    };
    
} // namespace lldb
