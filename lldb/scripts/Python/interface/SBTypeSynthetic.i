//===-- SWIG Interface for SBTypeSynthetic-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
    %feature("docstring",
    "Represents a summary that can be associated to one or more types.
    ") SBTypeSynthetic;
    
    class SBTypeSynthetic
    {
    public:
        
        SBTypeSynthetic();
        
        static lldb::SBTypeSynthetic
        CreateWithClassName (const char* data, uint32_t options = 0);
        
        static lldb::SBTypeSynthetic
        CreateWithScriptCode (const char* data, uint32_t options = 0);
        
        SBTypeSynthetic (const lldb::SBTypeSynthetic &rhs);
        
        ~SBTypeSynthetic ();
        
        bool
        IsValid() const;
        
        bool
        IsEqualTo (lldb::SBTypeSynthetic &rhs);
        
        bool
        IsClassCode();
        
        const char*
        GetData ();
        
        void
        SetClassName (const char* data);
        
        void
        SetClassCode (const char* data);

        uint32_t
        GetOptions ();
        
        void
        SetOptions (uint32_t);
        
        bool
        GetDescription (lldb::SBStream &description, 
                        lldb::DescriptionLevel description_level);
        
        bool
        operator == (lldb::SBTypeSynthetic &rhs);

        bool
        operator != (lldb::SBTypeSynthetic &rhs);
        
        %pythoncode %{
            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)
            
            __swig_getmethods__["contains_code"] = IsClassCode
            if _newclass: contains_code = property(IsClassCode, None)
            
            __swig_getmethods__["synthetic_data"] = GetData
            if _newclass: synthetic_data = property(GetData, None)
        %}
        
    };
    
} // namespace lldb
