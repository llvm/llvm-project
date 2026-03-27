STRING_EXTENSION_OUTSIDE(SBModuleSpecList)

%extend lldb::SBModuleSpecList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of ModuleSpec in a lldb.SBModuleSpecList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all ModuleSpecs in a lldb.SBModuleSpecList object.'''
      return lldb_iter(self, 'GetSize', 'GetSpecAtIndex')

    def __getitem__(self, key):
      '''Access module specs by index, full or partial path, or regular expression.
         specs[0]                      - access by integer index
         specs[-1]                     - access by negative integer index
         specs['a.out']                - find first spec matching file basename
         specs['/usr/lib/liba.dylib']  - find first spec matching file fullpath
         specs[re.compile(r'lib.*')]   - find all specs matching regex on fullpath
      '''
      count = len(self)
      if type(key) is int:
          if -count <= key < count:
              key %= count
              return self.GetSpecAtIndex(key)
          else:
              raise IndexError("list index out of range")
      elif type(key) is str:
          if key.find('/') == -1:
              for idx in range(count):
                  spec = self.GetSpecAtIndex(idx)
                  if spec.GetFileSpec().GetFilename() == key:
                      return spec
          else:
              for idx in range(count):
                  spec = self.GetSpecAtIndex(idx)
                  if str(spec.GetFileSpec()) == key:
                      return spec
          return None
      elif isinstance(key, type(re.compile(''))):
          matching_specs = []
          for idx in range(count):
              spec = self.GetSpecAtIndex(idx)
              re_match = key.search(str(spec.GetFileSpec()))
              if re_match:
                  matching_specs.append(spec)
          return matching_specs
      else:
          raise TypeError("unsupported index type: %s" % type(key))
    %}
#endif
}
