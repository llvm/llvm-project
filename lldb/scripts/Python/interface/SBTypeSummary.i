//===-- SWIG Interface for SBTypeSummary---------------------------*- C++ -*-===//
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
    ") SBTypeSummary;
    
    class SBTypeSummary
    {
    public:
        
        SBTypeSummary();
        
        static SBTypeSummary
        CreateWithSummaryString (const char* data, uint32_t options = 0);
        
        static SBTypeSummary
        CreateWithFunctionName (const char* data, uint32_t options = 0);
        
        static SBTypeSummary
        CreateWithScriptCode (const char* data, uint32_t options = 0);
        
        SBTypeSummary (const lldb::SBTypeSummary &rhs);
        
        ~SBTypeSummary ();
        
        bool
        IsValid() const;
        
        bool
        IsEqualTo (lldb::SBTypeSummary &rhs);
        
        bool
        IsFunctionCode();
        
        bool
        IsFunctionName();
        
        bool
        IsSummaryString();
        
        const char*
        GetData ();
        
        void
        SetSummaryString (const char* data);
        
        void
        SetFunctionName (const char* data);
        
        void
        SetFunctionCode (const char* data);
        
        uint32_t
        GetOptions ();

        void
        SetOptions (uint32_t);
        
        bool
        GetDescription (lldb::SBStream &description, 
                        lldb::DescriptionLevel description_level);
        
        bool
        operator == (lldb::SBTypeSummary &rhs);
        
        bool
        operator != (lldb::SBTypeSummary &rhs);
        
        %pythoncode %{
            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)
            
            __swig_getmethods__["is_summary_string"] = IsSummaryString
            if _newclass: is_summary_string = property(IsSummaryString, None)

            __swig_getmethods__["is_function_name"] = IsFunctionName
            if _newclass: is_function_name = property(IsFunctionName, None)

            __swig_getmethods__["is_function_name"] = IsFunctionCode
            if _newclass: is_function_name = property(IsFunctionCode, None)

            __swig_getmethods__["summary_data"] = GetData
            if _newclass: summary_data = property(GetData, None)
        %}
        
    };

} // namespace lldb

