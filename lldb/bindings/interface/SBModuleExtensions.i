#ifdef SWIGPYTHON
%pythoncode%{
# ==================================
# Helper function for SBModule class
# ==================================
def in_range(symbol, section):
    """Test whether a symbol is within the range of a section."""
    symSA = symbol.GetStartAddress().GetFileAddress()
    symEA = symbol.GetEndAddress().GetFileAddress()
    secSA = section.GetFileAddress()
    secEA = secSA + section.GetByteSize()

    if symEA != LLDB_INVALID_ADDRESS:
        if secSA <= symSA and symEA <= secEA:
            return True
        else:
            return False
    else:
        if secSA <= symSA and symSA < secEA:
            return True
        else:
            return False
%}
#endif

STRING_EXTENSION_OUTSIDE(SBModule)

%extend lldb::SBModule {
#ifdef SWIGPYTHON
    %pythoncode %{
        # operator== is a free function, which swig does not handle, so we inject
        # our own equality operator here
        def __eq__(self, other):
            return not self.__ne__(other)

        def __len__(self):
            '''Return the number of symbols in a lldb.SBModule object.'''
            return self.GetNumSymbols()

        def __iter__(self):
            '''Iterate over all symbols in a lldb.SBModule object.'''
            return lldb_iter(self, 'GetNumSymbols', 'GetSymbolAtIndex')

        def section_iter(self):
            '''Iterate over all sections in a lldb.SBModule object.'''
            return lldb_iter(self, 'GetNumSections', 'GetSectionAtIndex')

        def compile_unit_iter(self):
            '''Iterate over all compile units in a lldb.SBModule object.'''
            return lldb_iter(self, 'GetNumCompileUnits', 'GetCompileUnitAtIndex')

        def symbol_in_section_iter(self, section):
            '''Given a module and its contained section, returns an iterator on the
            symbols within the section.'''
            for sym in self:
                if in_range(sym, section):
                    yield sym

        class symbols_access(object):
            re_compile_type = type(re.compile('.'))
            '''A helper object that will lazily hand out lldb.SBSymbol objects for a module when supplied an index, name, or regular expression.'''
            def __init__(self, sbmodule):
                self.sbmodule = sbmodule

            def __len__(self):
                if self.sbmodule:
                    return int(self.sbmodule.GetNumSymbols())
                return 0

            def __getitem__(self, key):
                count = len(self)
                if type(key) is int:
                    if -count <= key < count:
                        key %= count
                        return self.sbmodule.GetSymbolAtIndex(key)
                elif type(key) is str:
                    matches = []
                    sc_list = self.sbmodule.FindSymbols(key)
                    for sc in sc_list:
                        symbol = sc.symbol
                        if symbol:
                            matches.append(symbol)
                    return matches
                elif isinstance(key, self.re_compile_type):
                    matches = []
                    for idx in range(count):
                        symbol = self.sbmodule.GetSymbolAtIndex(idx)
                        added = False
                        name = symbol.name
                        if name:
                            re_match = key.search(name)
                            if re_match:
                                matches.append(symbol)
                                added = True
                        if not added:
                            mangled = symbol.mangled
                            if mangled:
                                re_match = key.search(mangled)
                                if re_match:
                                    matches.append(symbol)
                    return matches
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        def get_symbols_access_object(self):
            '''An accessor function that returns a symbols_access() object which allows lazy symbol access from a lldb.SBModule object.'''
            return self.symbols_access (self)

        def get_compile_units_access_object (self):
            '''An accessor function that returns a compile_units_access() object which allows lazy compile unit access from a lldb.SBModule object.'''
            return self.compile_units_access (self)

        def get_symbols_array(self):
            '''An accessor function that returns a list() that contains all symbols in a lldb.SBModule object.'''
            symbols = []
            for idx in range(self.num_symbols):
                symbols.append(self.GetSymbolAtIndex(idx))
            return symbols

        class sections_access(object):
            re_compile_type = type(re.compile('.'))
            '''A helper object that will lazily hand out lldb.SBSection objects for a module when supplied an index, name, or regular expression.'''
            def __init__(self, sbmodule):
                self.sbmodule = sbmodule

            def __len__(self):
                if self.sbmodule:
                    return int(self.sbmodule.GetNumSections())
                return 0

            def __getitem__(self, key):
                count = len(self)
                if type(key) is int:
                    if -count <= key < count:
                        key %= count
                        return self.sbmodule.GetSectionAtIndex(key)
                elif type(key) is str:
                    for idx in range(count):
                        section = self.sbmodule.GetSectionAtIndex(idx)
                        if section.name == key:
                            return section
                elif isinstance(key, self.re_compile_type):
                    matches = []
                    for idx in range(count):
                        section = self.sbmodule.GetSectionAtIndex(idx)
                        name = section.name
                        if name:
                            re_match = key.search(name)
                            if re_match:
                                matches.append(section)
                    return matches
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        class compile_units_access(object):
            re_compile_type = type(re.compile('.'))
            '''A helper object that will lazily hand out lldb.SBCompileUnit objects for a module when supplied an index, full or partial path, or regular expression.'''
            def __init__(self, sbmodule):
                self.sbmodule = sbmodule

            def __len__(self):
                if self.sbmodule:
                    return int(self.sbmodule.GetNumCompileUnits())
                return 0

            def __getitem__(self, key):
                count = len(self)
                if type(key) is int:
                    if -count <= key < count:
                        key %= count
                        return self.sbmodule.GetCompileUnitAtIndex(key)
                elif type(key) is str:
                    is_full_path = key[0] == '/'
                    for idx in range(count):
                        comp_unit = self.sbmodule.GetCompileUnitAtIndex(idx)
                        if is_full_path:
                            if comp_unit.file.fullpath == key:
                                return comp_unit
                        else:
                            if comp_unit.file.basename == key:
                                return comp_unit
                elif isinstance(key, self.re_compile_type):
                    matches = []
                    for idx in range(count):
                        comp_unit = self.sbmodule.GetCompileUnitAtIndex(idx)
                        fullpath = comp_unit.file.fullpath
                        if fullpath:
                            re_match = key.search(fullpath)
                            if re_match:
                                matches.append(comp_unit)
                    return matches
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        def get_sections_access_object(self):
            '''An accessor function that returns a sections_access() object which allows lazy section array access.'''
            return self.sections_access (self)

        def get_sections_array(self):
            '''An accessor function that returns an array object that contains all sections in this module object.'''
            if not hasattr(self, 'sections_array'):
                self.sections_array = []
                for idx in range(self.num_sections):
                    self.sections_array.append(self.GetSectionAtIndex(idx))
            return self.sections_array

        def get_compile_units_array(self):
            '''An accessor function that returns an array object that contains all compile_units in this module object.'''
            if not hasattr(self, 'compile_units_array'):
                self.compile_units_array = []
                for idx in range(self.GetNumCompileUnits()):
                    self.compile_units_array.append(self.GetCompileUnitAtIndex(idx))
            return self.compile_units_array

        symbols = property(get_symbols_array, None, doc='''A read only property that returns a list() of lldb.SBSymbol objects contained in this module.''')
        symbol = property(get_symbols_access_object, None, doc='''A read only property that can be used to access symbols by index ("symbol = module.symbol[0]"), name ("symbols = module.symbol['main']"), or using a regular expression ("symbols = module.symbol[re.compile(...)]"). The return value is a single lldb.SBSymbol object for array access, and a list() of lldb.SBSymbol objects for name and regular expression access''')
        sections = property(get_sections_array, None, doc='''A read only property that returns a list() of lldb.SBSection objects contained in this module.''')
        compile_units = property(get_compile_units_array, None, doc='''A read only property that returns a list() of lldb.SBCompileUnit objects contained in this module.''')
        section = property(get_sections_access_object, None, doc='''A read only property that can be used to access symbols by index ("section = module.section[0]"), name ("sections = module.section[\'main\']"), or using a regular expression ("sections = module.section[re.compile(...)]"). The return value is a single lldb.SBSection object for array access, and a list() of lldb.SBSection objects for name and regular expression access''')
        section = property(get_sections_access_object, None, doc='''A read only property that can be used to access compile units by index ("compile_unit = module.compile_unit[0]"), name ("compile_unit = module.compile_unit[\'main.cpp\']"), or using a regular expression ("compile_unit = module.compile_unit[re.compile(...)]"). The return value is a single lldb.SBCompileUnit object for array access or by full or partial path, and a list() of lldb.SBCompileUnit objects regular expressions.''')

        def get_uuid(self):
            return uuid.UUID (self.GetUUIDString())

        uuid = property(get_uuid, None, doc='''A read only property that returns a standard python uuid.UUID object that represents the UUID of this module.''')
        file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this object file for this module as it is represented where it is being debugged.''')
        platform_file = property(GetPlatformFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this object file for this module as it is represented on the current host system.''')
        byte_order = property(GetByteOrder, None, doc='''A read only property that returns an lldb enumeration value (lldb.eByteOrderLittle, lldb.eByteOrderBig, lldb.eByteOrderInvalid) that represents the byte order for this module.''')
        addr_size = property(GetAddressByteSize, None, doc='''A read only property that returns the size in bytes of an address for this module.''')
        triple = property(GetTriple, None, doc='''A read only property that returns the target triple (arch-vendor-os) for this module.''')
        num_symbols = property(GetNumSymbols, None, doc='''A read only property that returns number of symbols in the module symbol table as an integer.''')
        num_sections = property(GetNumSections, None, doc='''A read only property that returns number of sections in the module as an integer.''')

    %}
#endif
}
