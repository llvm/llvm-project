//===-- SWIG Interface for SBTypeSummary---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {
    class SBTypeSummaryOptions
    {
    public:
        SBTypeSummaryOptions();
        
        SBTypeSummaryOptions (const lldb::SBTypeSummaryOptions &rhs);
        
        ~SBTypeSummaryOptions ();
        
        bool
        IsValid ();
        
        lldb::LanguageType
        GetLanguage ();
        
        lldb::TypeSummaryCapping
        GetCapping ();
        
        void
        SetLanguage (lldb::LanguageType);
        
        void
        SetCapping (lldb::TypeSummaryCapping);
    };

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

