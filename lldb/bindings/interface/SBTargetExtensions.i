#ifdef SWIGPYTHON
%typemap(in) (const char **symbol_name, uint32_t num_names) {
  using namespace lldb_private;
  /* Check if is a list  */
  if (PythonList::Check($input)) {
    PythonList list(PyRefType::Borrowed, $input);
    $2 = list.GetSize();
    int i = 0;
    $1 = (char**)malloc(($2+1)*sizeof(char*));
    for (i = 0; i < $2; i++) {
      PythonString py_str = list.GetItemAtIndex(i).AsType<PythonString>();
      if (!py_str.IsAllocated()) {
        PyErr_SetString(PyExc_TypeError,"list must contain strings and blubby");
        free($1);
        return nullptr;
      }

      $1[i] = const_cast<char*>(py_str.GetString().data());
    }
    $1[i] = 0;
  } else if ($input == Py_None) {
    $1 =  NULL;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}
#endif

STRING_EXTENSION_LEVEL_OUTSIDE(SBTarget, lldb::eDescriptionLevelBrief)

%extend lldb::SBTarget {
#ifdef SWIGPYTHON
    %pythoncode %{
        class modules_access(object):
            '''A helper object that will lazily hand out lldb.SBModule objects for a target when supplied an index, or by full or partial path.'''
            def __init__(self, sbtarget):
                self.sbtarget = sbtarget

            def __len__(self):
                if self.sbtarget:
                    return int(self.sbtarget.GetNumModules())
                return 0

            def __getitem__(self, key):
                num_modules = self.sbtarget.GetNumModules()
                if type(key) is int:
                    if -num_modules <= key < num_modules:
                        key %= num_modules
                        return self.sbtarget.GetModuleAtIndex(key)
                elif type(key) is str:
                    if key.find('/') == -1:
                        for idx in range(num_modules):
                            module = self.sbtarget.GetModuleAtIndex(idx)
                            if module.file.basename == key:
                                return module
                    else:
                        for idx in range(num_modules):
                            module = self.sbtarget.GetModuleAtIndex(idx)
                            if module.file.fullpath == key:
                                return module
                    # See if the string is a UUID
                    try:
                        the_uuid = uuid.UUID(key)
                        if the_uuid:
                            for idx in range(num_modules):
                                module = self.sbtarget.GetModuleAtIndex(idx)
                                if module.uuid == the_uuid:
                                    return module
                    except:
                        return None
                elif type(key) is uuid.UUID:
                    for idx in range(num_modules):
                        module = self.sbtarget.GetModuleAtIndex(idx)
                        if module.uuid == key:
                            return module
                elif type(key) is re.SRE_Pattern:
                    matching_modules = []
                    for idx in range(num_modules):
                        module = self.sbtarget.GetModuleAtIndex(idx)
                        re_match = key.search(module.path.fullpath)
                        if re_match:
                            matching_modules.append(module)
                    return matching_modules
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        def get_modules_access_object(self):
            '''An accessor function that returns a modules_access() object which allows lazy module access from a lldb.SBTarget object.'''
            return self.modules_access(self)

        def get_modules_array(self):
            '''An accessor function that returns a list() that contains all modules in a lldb.SBTarget object.'''
            modules = []
            for idx in range(self.GetNumModules()):
                modules.append(self.GetModuleAtIndex(idx))
            return modules

        def module_iter(self):
            '''Returns an iterator over all modules in a lldb.SBTarget
            object.'''
            return lldb_iter(self, 'GetNumModules', 'GetModuleAtIndex')

        def breakpoint_iter(self):
            '''Returns an iterator over all breakpoints in a lldb.SBTarget
            object.'''
            return lldb_iter(self, 'GetNumBreakpoints', 'GetBreakpointAtIndex')

        class bkpts_access(object):
            '''A helper object that will lazily hand out bkpts for a target when supplied an index.'''
            def __init__(self, sbtarget):
                self.sbtarget = sbtarget

            def __len__(self):
                if self.sbtarget:
                    return int(self.sbtarget.GetNumBreakpoints())
                return 0

            def __getitem__(self, key):
                if isinstance(key, int):
                    count = len(self)
                    if -count <= key < count:
                        key %= count
                        return self.sbtarget.GetBreakpointAtIndex(key)
                return None

        def get_bkpts_access_object(self):
            '''An accessor function that returns a bkpts_access() object which allows lazy bkpt access from a lldb.SBtarget object.'''
            return self.bkpts_access(self)

        def get_target_bkpts(self):
            '''An accessor function that returns a list() that contains all bkpts in a lldb.SBtarget object.'''
            bkpts = []
            for idx in range(self.GetNumBreakpoints()):
                bkpts.append(self.GetBreakpointAtIndex(idx))
            return bkpts

        def watchpoint_iter(self):
            '''Returns an iterator over all watchpoints in a lldb.SBTarget
            object.'''
            return lldb_iter(self, 'GetNumWatchpoints', 'GetWatchpointAtIndex')

        class watchpoints_access(object):
            '''A helper object that will lazily hand out watchpoints for a target when supplied an index.'''
            def __init__(self, sbtarget):
                self.sbtarget = sbtarget

            def __len__(self):
                if self.sbtarget:
                    return int(self.sbtarget.GetNumWatchpoints())
                return 0

            def __getitem__(self, key):
                if isinstance(key, int):
                    count = len(self)
                    if -count <= key < count:
                        key %= count
                        return self.sbtarget.GetWatchpointAtIndex(key)
                return None

        def get_watchpoints_access_object(self):
            '''An accessor function that returns a watchpoints_access() object which allows lazy watchpoint access from a lldb.SBtarget object.'''
            return self.watchpoints_access(self)

        def get_target_watchpoints(self):
            '''An accessor function that returns a list() that contains all watchpoints in a lldb.SBtarget object.'''
            watchpoints = []
            for idx in range(self.GetNumWatchpoints()):
                bkpts.append(self.GetWatchpointAtIndex(idx))
            return watchpoints

        modules = property(get_modules_array, None, doc='''A read only property that returns a list() of lldb.SBModule objects contained in this target. This list is a list all modules that the target currently is tracking (the main executable and all dependent shared libraries).''')
        module = property(get_modules_access_object, None, doc=r'''A read only property that returns an object that implements python operator overloading with the square brackets().\n    target.module[<int>] allows array access to any modules.\n    target.module[<str>] allows access to modules by basename, full path, or uuid string value.\n    target.module[uuid.UUID()] allows module access by UUID.\n    target.module[re] allows module access using a regular expression that matches the module full path.''')
        process = property(GetProcess, None, doc='''A read only property that returns an lldb object that represents the process (lldb.SBProcess) that this target owns.''')
        executable = property(GetExecutable, None, doc='''A read only property that returns an lldb object that represents the main executable module (lldb.SBModule) for this target.''')
        debugger = property(GetDebugger, None, doc='''A read only property that returns an lldb object that represents the debugger (lldb.SBDebugger) that owns this target.''')
        num_breakpoints = property(GetNumBreakpoints, None, doc='''A read only property that returns the number of breakpoints that this target has as an integer.''')
        breakpoints = property(get_target_bkpts, None, doc='''A read only property that returns a list() of lldb.SBBreakpoint objects for all breakpoints in this target.''')
        breakpoint = property(get_bkpts_access_object, None, doc='''A read only property that returns an object that can be used to access breakpoints as an array ("bkpt_12 = lldb.target.bkpt[12]").''')
        num_watchpoints = property(GetNumWatchpoints, None, doc='''A read only property that returns the number of watchpoints that this target has as an integer.''')
        watchpoints = property(get_target_watchpoints, None, doc='''A read only property that returns a list() of lldb.SBwatchpoint objects for all watchpoints in this target.''')
        watchpoint = property(get_watchpoints_access_object, None, doc='''A read only property that returns an object that can be used to access watchpoints as an array ("watchpoint_12 = lldb.target.watchpoint[12]").''')
        broadcaster = property(GetBroadcaster, None, doc='''A read only property that an lldb object that represents the broadcaster (lldb.SBBroadcaster) for this target.''')
        byte_order = property(GetByteOrder, None, doc='''A read only property that returns an lldb enumeration value (lldb.eByteOrderLittle, lldb.eByteOrderBig, lldb.eByteOrderInvalid) that represents the byte order for this target.''')
        addr_size = property(GetAddressByteSize, None, doc='''A read only property that returns the size in bytes of an address for this target.''')
        triple = property(GetTriple, None, doc='''A read only property that returns the target triple (arch-vendor-os) for this target as a string.''')
        data_byte_size = property(GetDataByteSize, None, doc='''A read only property that returns the size in host bytes of a byte in the data address space for this target.''')
        code_byte_size = property(GetCodeByteSize, None, doc='''A read only property that returns the size in host bytes of a byte in the code address space for this target.''')
        platform = property(GetPlatform, None, doc='''A read only property that returns the platform associated with with this target.''')
    %}
#endif
}
