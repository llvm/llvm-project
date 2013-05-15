//===-- SWIG Interface for SBTypeFormat----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
    %feature("docstring",
             "Represents a format that can be associated to one or more types.
             ") SBTypeFormat;
    
    class SBTypeFormat
    {
    public:
        
        SBTypeFormat();
        
        SBTypeFormat (lldb::Format format, uint32_t options = 0);
        
        SBTypeFormat (const lldb::SBTypeFormat &rhs);
        
        ~SBTypeFormat ();
        
        bool
        IsValid() const;
        
        bool
        IsEqualTo (lldb::SBTypeFormat &rhs);
        
        lldb::Format
        GetFormat ();
        
        uint32_t
        GetOptions();
        
        void
        SetFormat (lldb::Format);
        
        void
        SetOptions (uint32_t);        
        
        bool
        GetDescription (lldb::SBStream &description, 
                        lldb::DescriptionLevel description_level);
        
        bool
        operator == (lldb::SBTypeFormat &rhs);

        bool
        operator != (lldb::SBTypeFormat &rhs);
        
        %pythoncode %{
            __swig_getmethods__["format"] = GetFormat
            __swig_setmethods__["format"] = SetFormat
            if _newclass: format = property(GetFormat, SetFormat)
            
            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)            
        %}

    };
    
    
} // namespace lldb

