STRING_EXTENSION_LEVEL_OUTSIDE(SBTypeCategory, lldb::eDescriptionLevelBrief)

%extend lldb::SBTypeCategory {
#ifdef SWIGPYTHON
        %pythoncode %{
            def __eq__(self, other):
                return not self.__ne__(other)

            def __int__(self):
                pass

            def __hex__(self):
                pass

            def __oct__(self):
                pass

            def __len__(self):
                pass

            def __iter__(self):
                pass

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
                        if -num_items <= key < num_items:
                            key %= num_items
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

            formats = property(get_formats_array, None, doc='''A read only property that returns a list() of lldb.SBTypeFormat objects contained in this category''')
            format = property(get_formats_access_object, None, doc=r'''A read only property that returns an object that you can use to look for formats by index or type name.''')
            summaries = property(get_summaries_array, None, doc='''A read only property that returns a list() of lldb.SBTypeSummary objects contained in this category''')
            summary = property(get_summaries_access_object, None, doc=r'''A read only property that returns an object that you can use to look for summaries by index or type name or regular expression.''')
            filters = property(get_filters_array, None, doc='''A read only property that returns a list() of lldb.SBTypeFilter objects contained in this category''')
            filter = property(get_filters_access_object, None, doc=r'''A read only property that returns an object that you can use to look for filters by index or type name or regular expression.''')
            synthetics = property(get_synthetics_array, None, doc='''A read only property that returns a list() of lldb.SBTypeSynthetic objects contained in this category''')
            synthetic = property(get_synthetics_access_object, None, doc=r'''A read only property that returns an object that you can use to look for synthetic children provider by index or type name or regular expression.''')
            num_formats = property(GetNumFormats, None)
            num_summaries = property(GetNumSummaries, None)
            num_filters = property(GetNumFilters, None)
            num_synthetics = property(GetNumSynthetics, None)
            name = property(GetName, None)
            enabled = property(GetEnabled, SetEnabled)
        %}
#endif
}
