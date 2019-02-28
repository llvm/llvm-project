//===-- SWIG Interface for SBModule -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

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

%feature("docstring",
"Represents an executable image and its associated object and symbol files.

The module is designed to be able to select a single slice of an
executable image as it would appear on disk and during program
execution.

You can retrieve SBModule from SBSymbolContext, which in turn is available
from SBFrame.

SBModule supports symbol iteration, for example,

    for symbol in module:
        name = symbol.GetName()
        saddr = symbol.GetStartAddress()
        eaddr = symbol.GetEndAddress()

and rich comparison methods which allow the API program to use,

    if thisModule == thatModule:
        print('This module is the same as that module')

to test module equality.  A module also contains object file sections, namely
SBSection.  SBModule supports section iteration through section_iter(), for
example,

    print('Number of sections: %d' % module.GetNumSections())
    for sec in module.section_iter():
        print(sec)

And to iterate the symbols within a SBSection, use symbol_in_section_iter(),

    # Iterates the text section and prints each symbols within each sub-section.
    for subsec in text_sec:
        print(INDENT + repr(subsec))
        for sym in exe_module.symbol_in_section_iter(subsec):
            print(INDENT2 + repr(sym))
            print(INDENT2 + 'symbol type: %s' % symbol_type_to_str(sym.GetType()))

produces this following output:

    [0x0000000100001780-0x0000000100001d5c) a.out.__TEXT.__text
        id = {0x00000004}, name = 'mask_access(MaskAction, unsigned int)', range = [0x00000001000017c0-0x0000000100001870)
        symbol type: code
        id = {0x00000008}, name = 'thread_func(void*)', range = [0x0000000100001870-0x00000001000019b0)
        symbol type: code
        id = {0x0000000c}, name = 'main', range = [0x00000001000019b0-0x0000000100001d5c)
        symbol type: code
        id = {0x00000023}, name = 'start', address = 0x0000000100001780
        symbol type: code
    [0x0000000100001d5c-0x0000000100001da4) a.out.__TEXT.__stubs
        id = {0x00000024}, name = '__stack_chk_fail', range = [0x0000000100001d5c-0x0000000100001d62)
        symbol type: trampoline
        id = {0x00000028}, name = 'exit', range = [0x0000000100001d62-0x0000000100001d68)
        symbol type: trampoline
        id = {0x00000029}, name = 'fflush', range = [0x0000000100001d68-0x0000000100001d6e)
        symbol type: trampoline
        id = {0x0000002a}, name = 'fgets', range = [0x0000000100001d6e-0x0000000100001d74)
        symbol type: trampoline
        id = {0x0000002b}, name = 'printf', range = [0x0000000100001d74-0x0000000100001d7a)
        symbol type: trampoline
        id = {0x0000002c}, name = 'pthread_create', range = [0x0000000100001d7a-0x0000000100001d80)
        symbol type: trampoline
        id = {0x0000002d}, name = 'pthread_join', range = [0x0000000100001d80-0x0000000100001d86)
        symbol type: trampoline
        id = {0x0000002e}, name = 'pthread_mutex_lock', range = [0x0000000100001d86-0x0000000100001d8c)
        symbol type: trampoline
        id = {0x0000002f}, name = 'pthread_mutex_unlock', range = [0x0000000100001d8c-0x0000000100001d92)
        symbol type: trampoline
        id = {0x00000030}, name = 'rand', range = [0x0000000100001d92-0x0000000100001d98)
        symbol type: trampoline
        id = {0x00000031}, name = 'strtoul', range = [0x0000000100001d98-0x0000000100001d9e)
        symbol type: trampoline
        id = {0x00000032}, name = 'usleep', range = [0x0000000100001d9e-0x0000000100001da4)
        symbol type: trampoline
    [0x0000000100001da4-0x0000000100001e2c) a.out.__TEXT.__stub_helper
    [0x0000000100001e2c-0x0000000100001f10) a.out.__TEXT.__cstring
    [0x0000000100001f10-0x0000000100001f68) a.out.__TEXT.__unwind_info
    [0x0000000100001f68-0x0000000100001ff8) a.out.__TEXT.__eh_frame
"
) SBModule;
class SBModule
{
public:

    SBModule ();

    SBModule (const lldb::SBModule &rhs);
     
    SBModule (const lldb::SBModuleSpec &module_spec);
    
    SBModule (lldb::SBProcess &process, 
              lldb::addr_t header_addr);

    ~SBModule ();

    bool
    IsValid () const;

    void
    Clear();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get const accessor for the module file specification.
    ///
    /// This function returns the file for the module on the host system
    /// that is running LLDB. This can differ from the path on the 
    /// platform since we might be doing remote debugging.
    ///
    /// @return
    ///     A const reference to the file specification object.
    //------------------------------------------------------------------
    ") GetFileSpec;
    lldb::SBFileSpec
    GetFileSpec () const;

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get accessor for the module platform file specification.
    ///
    /// Platform file refers to the path of the module as it is known on
    /// the remote system on which it is being debugged. For local 
    /// debugging this is always the same as Module::GetFileSpec(). But
    /// remote debugging might mention a file '/usr/lib/liba.dylib'
    /// which might be locally downloaded and cached. In this case the
    /// platform file could be something like:
    /// '/tmp/lldb/platform-cache/remote.host.computer/usr/lib/liba.dylib'
    /// The file could also be cached in a local developer kit directory.
    ///
    /// @return
    ///     A const reference to the file specification object.
    //------------------------------------------------------------------
    ") GetPlatformFileSpec;
    lldb::SBFileSpec
    GetPlatformFileSpec () const;

    bool
    SetPlatformFileSpec (const lldb::SBFileSpec &platform_file);
             
    lldb::SBFileSpec
    GetRemoteInstallFileSpec ();

    bool
    SetRemoteInstallFileSpec (lldb::SBFileSpec &file);

    %feature("docstring", "Returns the UUID of the module as a Python string."
    ) GetUUIDString;
    const char *
    GetUUIDString () const;

    lldb::SBSection
    FindSection (const char *sect_name);

    lldb::SBAddress
    ResolveFileAddress (lldb::addr_t vm_addr);

    lldb::SBSymbolContext
    ResolveSymbolContextForAddress (const lldb::SBAddress& addr, 
                                    uint32_t resolve_scope);

    bool
    GetDescription (lldb::SBStream &description);

    uint32_t
    GetNumCompileUnits();

    lldb::SBCompileUnit
    GetCompileUnitAtIndex (uint32_t);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find compile units related to *this module and passed source
    /// file.
    ///
    /// @param[in] sb_file_spec
    ///     A lldb::SBFileSpec object that contains source file
    ///     specification.
    ///
    /// @return
    ///     A lldb::SBSymbolContextList that gets filled in with all of
    ///     the symbol contexts for all the matches.
    //------------------------------------------------------------------
    ") FindCompileUnits;
    lldb::SBSymbolContextList
    FindCompileUnits (const lldb::SBFileSpec &sb_file_spec);

    size_t
    GetNumSymbols ();
    
    lldb::SBSymbol
    GetSymbolAtIndex (size_t idx);

    lldb::SBSymbol
    FindSymbol (const char *name,
                lldb::SymbolType type = eSymbolTypeAny);

    lldb::SBSymbolContextList
    FindSymbols (const char *name,
                 lldb::SymbolType type = eSymbolTypeAny);
             

    size_t
    GetNumSections ();

    lldb::SBSection
    GetSectionAtIndex (size_t idx);


    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find functions by name.
    ///
    /// @param[in] name
    ///     The name of the function we are looking for.
    ///
    /// @param[in] name_type_mask
    ///     A logical OR of one or more FunctionNameType enum bits that
    ///     indicate what kind of names should be used when doing the
    ///     lookup. Bits include fully qualified names, base names,
    ///     C++ methods, or ObjC selectors. 
    ///     See FunctionNameType for more details.
    ///
    /// @return
    ///     A symbol context list that gets filled in with all of the
    ///     matches.
    //------------------------------------------------------------------
    ") FindFunctions;
    lldb::SBSymbolContextList
    FindFunctions (const char *name, 
                   uint32_t name_type_mask = lldb::eFunctionNameTypeAny);
    
    lldb::SBType
    FindFirstType (const char* name);

    lldb::SBTypeList
    FindTypes (const char* type);

    lldb::SBType
    GetTypeByID (lldb::user_id_t uid);

    lldb::SBType
    GetBasicType(lldb::BasicType type);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get all types matching \a type_mask from debug info in this
    /// module.
    ///
    /// @param[in] type_mask
    ///     A bitfield that consists of one or more bits logically OR'ed
    ///     together from the lldb::TypeClass enumeration. This allows
    ///     you to request only structure types, or only class, struct
    ///     and union types. Passing in lldb::eTypeClassAny will return
    ///     all types found in the debug information for this module.
    ///
    /// @return
    ///     A list of types in this module that match \a type_mask
    //------------------------------------------------------------------
    ") GetTypes;
    lldb::SBTypeList
    GetTypes (uint32_t type_mask = lldb::eTypeClassAny);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find global and static variables by name.
    ///
    /// @param[in] target
    ///     A valid SBTarget instance representing the debuggee.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @param[in] max_matches
    ///     Allow the number of matches to be limited to \a max_matches.
    ///
    /// @return
    ///     A list of matched variables in an SBValueList.
    //------------------------------------------------------------------
    ") FindGlobalVariables;
    lldb::SBValueList
    FindGlobalVariables (lldb::SBTarget &target, 
                         const char *name, 
                         uint32_t max_matches);
    
    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find the first global (or static) variable by name.
    ///
    /// @param[in] target
    ///     A valid SBTarget instance representing the debuggee.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @return
    ///     An SBValue that gets filled in with the found variable (if any).
    //------------------------------------------------------------------
    ") FindFirstGlobalVariable;
    lldb::SBValue
    FindFirstGlobalVariable (lldb::SBTarget &target, const char *name);
             
    lldb::ByteOrder
    GetByteOrder ();
    
    uint32_t
    GetAddressByteSize();
    
    const char *
    GetTriple ();
    
    uint32_t
    GetVersion (uint32_t *versions, 
                uint32_t num_versions);

    lldb::SBFileSpec
    GetSymbolFileSpec() const;

    lldb::SBAddress
    GetObjectFileHeaderAddress() const;

    lldb::SBAddress
    GetObjectFileEntryPointAddress() const;

    bool
    operator == (const lldb::SBModule &rhs) const;
             
    bool
    operator != (const lldb::SBModule &rhs) const;
             
    %pythoncode %{
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
                    if key < count:
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
                    if key < count:
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
                    if key < count:
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

        __swig_getmethods__["symbols"] = get_symbols_array
        if _newclass: symbols = property(get_symbols_array, None, doc='''A read only property that returns a list() of lldb.SBSymbol objects contained in this module.''')

        __swig_getmethods__["symbol"] = get_symbols_access_object
        if _newclass: symbol = property(get_symbols_access_object, None, doc='''A read only property that can be used to access symbols by index ("symbol = module.symbol[0]"), name ("symbols = module.symbol['main']"), or using a regular expression ("symbols = module.symbol[re.compile(...)]"). The return value is a single lldb.SBSymbol object for array access, and a list() of lldb.SBSymbol objects for name and regular expression access''')

        __swig_getmethods__["sections"] = get_sections_array
        if _newclass: sections = property(get_sections_array, None, doc='''A read only property that returns a list() of lldb.SBSection objects contained in this module.''')

        __swig_getmethods__["compile_units"] = get_compile_units_array
        if _newclass: compile_units = property(get_compile_units_array, None, doc='''A read only property that returns a list() of lldb.SBCompileUnit objects contained in this module.''')

        __swig_getmethods__["section"] = get_sections_access_object
        if _newclass: section = property(get_sections_access_object, None, doc='''A read only property that can be used to access symbols by index ("section = module.section[0]"), name ("sections = module.section[\'main\']"), or using a regular expression ("sections = module.section[re.compile(...)]"). The return value is a single lldb.SBSection object for array access, and a list() of lldb.SBSection objects for name and regular expression access''')

        __swig_getmethods__["compile_unit"] = get_compile_units_access_object
        if _newclass: section = property(get_sections_access_object, None, doc='''A read only property that can be used to access compile units by index ("compile_unit = module.compile_unit[0]"), name ("compile_unit = module.compile_unit[\'main.cpp\']"), or using a regular expression ("compile_unit = module.compile_unit[re.compile(...)]"). The return value is a single lldb.SBCompileUnit object for array access or by full or partial path, and a list() of lldb.SBCompileUnit objects regular expressions.''')

        def get_uuid(self):
            return uuid.UUID (self.GetUUIDString())
        
        __swig_getmethods__["uuid"] = get_uuid
        if _newclass: uuid = property(get_uuid, None, doc='''A read only property that returns a standard python uuid.UUID object that represents the UUID of this module.''')
        
        __swig_getmethods__["file"] = GetFileSpec
        if _newclass: file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this object file for this module as it is represented where it is being debugged.''')
        
        __swig_getmethods__["platform_file"] = GetPlatformFileSpec
        if _newclass: platform_file = property(GetPlatformFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this object file for this module as it is represented on the current host system.''')
        
        __swig_getmethods__["byte_order"] = GetByteOrder
        if _newclass: byte_order = property(GetByteOrder, None, doc='''A read only property that returns an lldb enumeration value (lldb.eByteOrderLittle, lldb.eByteOrderBig, lldb.eByteOrderInvalid) that represents the byte order for this module.''')
        
        __swig_getmethods__["addr_size"] = GetAddressByteSize
        if _newclass: addr_size = property(GetAddressByteSize, None, doc='''A read only property that returns the size in bytes of an address for this module.''')
        
        __swig_getmethods__["triple"] = GetTriple
        if _newclass: triple = property(GetTriple, None, doc='''A read only property that returns the target triple (arch-vendor-os) for this module.''')

        __swig_getmethods__["num_symbols"] = GetNumSymbols
        if _newclass: num_symbols = property(GetNumSymbols, None, doc='''A read only property that returns number of symbols in the module symbol table as an integer.''')
        
        __swig_getmethods__["num_sections"] = GetNumSections
        if _newclass: num_sections = property(GetNumSections, None, doc='''A read only property that returns number of sections in the module as an integer.''')
        
    %}

};

} // namespace lldb
