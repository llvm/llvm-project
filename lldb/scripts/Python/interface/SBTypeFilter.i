//===-- SWIG Interface for SBTypeFilter----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {
    
    %feature("docstring",
    "Represents a filter that can be associated to one or more types.
    ") SBTypeFilter;
    
    class SBTypeFilter
    {
        public:
        
        SBTypeFilter();
        
        SBTypeFilter (uint32_t options);
        
        SBTypeFilter (const lldb::SBTypeFilter &rhs);
        
        ~SBTypeFilter ();
        
        bool
        IsValid() const;
        
        bool
        IsEqualTo (lldb::SBTypeFilter &rhs);
        
        uint32_t
        GetNumberOfExpressionPaths ();
        
        const char*
        GetExpressionPathAtIndex (uint32_t i);
        
        bool
        ReplaceExpressionPathAtIndex (uint32_t i, const char* item);
        
        void
        AppendExpressionPath (const char* item);
        
        void
        Clear();
        
        uint32_t
        GetOptions();
        
        void
        SetOptions (uint32_t);
        
        bool
        GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level);
        
        bool
        operator == (lldb::SBTypeFilter &rhs);
        
        bool
        operator != (lldb::SBTypeFilter &rhs);
        
        %pythoncode %{
            __swig_getmethods__["options"] = GetOptions
            __swig_setmethods__["options"] = SetOptions
            if _newclass: options = property(GetOptions, SetOptions)
            
            __swig_getmethods__["count"] = GetNumberOfExpressionPaths
            if _newclass: count = property(GetNumberOfExpressionPaths, None)
        %}
                
    };

} // namespace lldb
