STRING_EXTENSION_OUTSIDE(SBBlock)

%extend lldb::SBBlock {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __int__(self):
            pass

        def __len__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __iter__(self):
            pass

        def get_range_at_index(self, idx):
            if idx < self.GetNumRanges():
                return [self.GetRangeStartAddress(idx), self.GetRangeEndAddress(idx)]
            return []

        class ranges_access(object):
            '''A helper object that will lazily hand out an array of lldb.SBAddress that represent address ranges for a block.'''
            def __init__(self, sbblock):
                self.sbblock = sbblock

            def __len__(self):
                if self.sbblock:
                    return int(self.sbblock.GetNumRanges())
                return 0

            def __getitem__(self, key):
                count = len(self)
                if type(key) is int:
                    return self.sbblock.get_range_at_index (key);
                if isinstance(key, SBAddress):
                    range_idx = self.sbblock.GetRangeIndexForBlockAddress(key);
                    if range_idx < len(self):
                        return [self.sbblock.GetRangeStartAddress(range_idx), self.sbblock.GetRangeEndAddress(range_idx)]
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        def get_ranges_access_object(self):
            '''An accessor function that returns a ranges_access() object which allows lazy block address ranges access.'''
            return self.ranges_access (self)

        def get_ranges_array(self):
            '''An accessor function that returns an array object that contains all ranges in this block object.'''
            if not hasattr(self, 'ranges_array'):
                self.ranges_array = []
                for idx in range(self.num_ranges):
                    self.ranges_array.append ([self.GetRangeStartAddress(idx), self.GetRangeEndAddress(idx)])
            return self.ranges_array

        def get_call_site(self):
            return declaration(self.GetInlinedCallSiteFile(), self.GetInlinedCallSiteLine(), self.GetInlinedCallSiteColumn())

        parent = property(GetParent, None, doc='''A read only property that returns the same result as GetParent().''')
        first_child = property(GetFirstChild, None, doc='''A read only property that returns the same result as GetFirstChild().''')
        call_site = property(get_call_site, None, doc='''A read only property that returns a lldb.declaration object that contains the inlined call site file, line and column.''')
        sibling = property(GetSibling, None, doc='''A read only property that returns the same result as GetSibling().''')
        name = property(GetInlinedName, None, doc='''A read only property that returns the same result as GetInlinedName().''')
        inlined_block = property(GetContainingInlinedBlock, None, doc='''A read only property that returns the same result as GetContainingInlinedBlock().''')
        range = property(get_ranges_access_object, None, doc='''A read only property that allows item access to the address ranges for a block by integer (range = block.range[0]) and by lldb.SBAddress (find the range that contains the specified lldb.SBAddress like "pc_range = lldb.frame.block.range[frame.addr]").''')
        ranges = property(get_ranges_array, None, doc='''A read only property that returns a list() object that contains all of the address ranges for the block.''')
        num_ranges = property(GetNumRanges, None, doc='''A read only property that returns the same result as GetNumRanges().''')
    %}
#endif
}
