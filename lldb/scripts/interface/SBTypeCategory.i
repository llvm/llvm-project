//===-- SWIG Interface for SBTypeCategory---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

        explicit operator bool() const;
        
        bool
        GetEnabled ();
        
        void
        SetEnabled (bool);
        
        const char*
        GetName();
        
        lldb::LanguageType
        GetLanguageAtIndex (uint32_t idx);
        
        uint32_t
        GetNumLanguages ();
        
        void
        AddLanguage (lldb::LanguageType language);
        
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
            
            class formatters_access_class(object):
                '''A helper object that will lazily hand out formatters for a specific category.'''
                def __init__(self, sbcategory, get_count_function, get_at_index_function, get_by_name_function):
                    self.sbcategory = sbcategory
                    self.get_count_function = get_count_function
                    self.get_at_index_function = get_at_index_function
                    self.get_by_name_function = get_by_name_function
                    self.regex_type = type(re.compile('.'))


                def __len__(self):
                    if self.sbcategory and self.get_count_function:
                        return int(self.get_count_function(self.sbcategory))
                    return 0

                def __getitem__(self, key):
                    num_items = len(self)
                    if type(key) is int:
                        if key < num_items:
                            return self.get_at_index_function(self.sbcategory,key)
                    elif type(key) is str:
                        return self.get_by_name_function(self.sbcategory,SBTypeNameSpecifier(key))
                    elif isinstance(key,self.regex_type):
                        return self.get_by_name_function(self.sbcategory,SBTypeNameSpecifier(key.pattern,True))
                    else:
                        print("error: unsupported item type: %s" % type(key))
                    return None

            def get_formats_access_object(self):
                '''An accessor function that returns an accessor object which allows lazy format access from a lldb.SBTypeCategory object.'''
                return self.formatters_access_class (self,self.__class__.GetNumFormats,self.__class__.GetFormatAtIndex,self.__class__.GetFormatForType)

            def get_formats_array(self):
                '''An accessor function that returns a list() that contains all formats in a lldb.SBCategory object.'''
                formats = []
                for idx in range(self.GetNumFormats()):
                    formats.append(self.GetFormatAtIndex(idx))
                return formats

            def get_summaries_access_object(self):
                '''An accessor function that returns an accessor object which allows lazy summary access from a lldb.SBTypeCategory object.'''
                return self.formatters_access_class (self,self.__class__.GetNumSummaries,self.__class__.GetSummaryAtIndex,self.__class__.GetSummaryForType)

            def get_summaries_array(self):
                '''An accessor function that returns a list() that contains all summaries in a lldb.SBCategory object.'''
                summaries = []
                for idx in range(self.GetNumSummaries()):
                    summaries.append(self.GetSummaryAtIndex(idx))
                return summaries

            def get_synthetics_access_object(self):
                '''An accessor function that returns an accessor object which allows lazy synthetic children provider access from a lldb.SBTypeCategory object.'''
                return self.formatters_access_class (self,self.__class__.GetNumSynthetics,self.__class__.GetSyntheticAtIndex,self.__class__.GetSyntheticForType)

            def get_synthetics_array(self):
                '''An accessor function that returns a list() that contains all synthetic children providers in a lldb.SBCategory object.'''
                synthetics = []
                for idx in range(self.GetNumSynthetics()):
                    synthetics.append(self.GetSyntheticAtIndex(idx))
                return synthetics

            def get_filters_access_object(self):
                '''An accessor function that returns an accessor object which allows lazy filter access from a lldb.SBTypeCategory object.'''
                return self.formatters_access_class (self,self.__class__.GetNumFilters,self.__class__.GetFilterAtIndex,self.__class__.GetFilterForType)

            def get_filters_array(self):
                '''An accessor function that returns a list() that contains all filters in a lldb.SBCategory object.'''
                filters = []
                for idx in range(self.GetNumFilters()):
                    filters.append(self.GetFilterAtIndex(idx))
                return filters

            __swig_getmethods__["formats"] = get_formats_array
            if _newclass: formats = property(get_formats_array, None, doc='''A read only property that returns a list() of lldb.SBTypeFormat objects contained in this category''')

            __swig_getmethods__["format"] = get_formats_access_object
            if _newclass: format = property(get_formats_access_object, None, doc=r'''A read only property that returns an object that you can use to look for formats by index or type name.''')

            __swig_getmethods__["summaries"] = get_summaries_array
            if _newclass: summaries = property(get_summaries_array, None, doc='''A read only property that returns a list() of lldb.SBTypeSummary objects contained in this category''')

            __swig_getmethods__["summary"] = get_summaries_access_object
            if _newclass: summary = property(get_summaries_access_object, None, doc=r'''A read only property that returns an object that you can use to look for summaries by index or type name or regular expression.''')

            __swig_getmethods__["filters"] = get_filters_array
            if _newclass: filters = property(get_filters_array, None, doc='''A read only property that returns a list() of lldb.SBTypeFilter objects contained in this category''')

            __swig_getmethods__["filter"] = get_filters_access_object
            if _newclass: filter = property(get_filters_access_object, None, doc=r'''A read only property that returns an object that you can use to look for filters by index or type name or regular expression.''')

            __swig_getmethods__["synthetics"] = get_synthetics_array
            if _newclass: synthetics = property(get_synthetics_array, None, doc='''A read only property that returns a list() of lldb.SBTypeSynthetic objects contained in this category''')

            __swig_getmethods__["synthetic"] = get_synthetics_access_object
            if _newclass: synthetic = property(get_synthetics_access_object, None, doc=r'''A read only property that returns an object that you can use to look for synthetic children provider by index or type name or regular expression.''')
            
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

