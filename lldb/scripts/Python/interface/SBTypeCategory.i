//===-- SWIG Interface for SBTypeCategory---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
    %feature("docstring",
    "Represents a category that can contain formatters for types.
    ") SBTypeCategory;
    
    class SBTypeCategory
    {
    public:
        
        SBTypeCategory();
        
        SBTypeCategory (const lldb::SBTypeCategory &rhs);
        
        ~SBTypeCategory ();
        
        bool
        IsValid() const;
        
        bool
        GetEnabled ();
        
        void
        SetEnabled (bool);
        
        const char*
        GetName();
        
        bool
        GetDescription (lldb::SBStream &description, 
                        lldb::DescriptionLevel description_level);
        
        uint32_t
        GetNumFormats ();
        
        uint32_t
        GetNumSummaries ();
        
        uint32_t
        GetNumFilters ();
        
        uint32_t
        GetNumSynthetics ();
        
        lldb::SBTypeNameSpecifier
        GetTypeNameSpecifierForFilterAtIndex (uint32_t);
        
        lldb::SBTypeNameSpecifier
        GetTypeNameSpecifierForFormatAtIndex (uint32_t);
        
        lldb::SBTypeNameSpecifier
        GetTypeNameSpecifierForSummaryAtIndex (uint32_t);

        lldb::SBTypeNameSpecifier
        GetTypeNameSpecifierForSyntheticAtIndex (uint32_t);
        
        lldb::SBTypeFilter
        GetFilterForType (lldb::SBTypeNameSpecifier);

        lldb::SBTypeFormat
        GetFormatForType (lldb::SBTypeNameSpecifier);
        
        lldb::SBTypeSummary
        GetSummaryForType (lldb::SBTypeNameSpecifier);

        lldb::SBTypeSynthetic
        GetSyntheticForType (lldb::SBTypeNameSpecifier);
        
        lldb::SBTypeFilter
        GetFilterAtIndex (uint32_t);
        
        lldb::SBTypeFormat
        GetFormatAtIndex (uint32_t);
        
        lldb::SBTypeSummary
        GetSummaryAtIndex (uint32_t);
        
        lldb::SBTypeSynthetic
        GetSyntheticAtIndex (uint32_t);
        
        bool
        AddTypeFormat (lldb::SBTypeNameSpecifier,
                       lldb::SBTypeFormat);
        
        bool
        DeleteTypeFormat (lldb::SBTypeNameSpecifier);
        
        bool
        AddTypeSummary (lldb::SBTypeNameSpecifier,
                        lldb::SBTypeSummary);
        
        bool
        DeleteTypeSummary (lldb::SBTypeNameSpecifier);
        
        bool
        AddTypeFilter (lldb::SBTypeNameSpecifier,
                       lldb::SBTypeFilter);
        
        bool
        DeleteTypeFilter (lldb::SBTypeNameSpecifier);
        
        bool
        AddTypeSynthetic (lldb::SBTypeNameSpecifier,
                          lldb::SBTypeSynthetic);
        
        bool
        DeleteTypeSynthetic (lldb::SBTypeNameSpecifier);
        
        %pythoncode %{
            __swig_getmethods__["num_formats"] = GetNumFormats
            if _newclass: num_formats = property(GetNumFormats, None)
            __swig_getmethods__["num_summaries"] = GetNumSummaries
            if _newclass: num_summaries = property(GetNumSummaries, None)
            __swig_getmethods__["num_filters"] = GetNumFilters
            if _newclass: num_filters = property(GetNumFilters, None)
            __swig_getmethods__["num_synthetics"] = GetNumSynthetics
            if _newclass: num_synthetics = property(GetNumSynthetics, None)
            
            __swig_getmethods__["name"] = GetName
            if _newclass: name = property(GetName, None)
            
            __swig_getmethods__["enabled"] = GetEnabled
            __swig_setmethods__["enabled"] = SetEnabled
            if _newclass: enabled = property(GetEnabled, SetEnabled)
        %}

    };

    
} // namespace lldb

