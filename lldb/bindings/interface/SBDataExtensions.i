STRING_EXTENSION_OUTSIDE(SBData)

%extend lldb::SBData {
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
            return self.GetByteSize()

        def __iter__(self):
            pass

        class read_data_helper:
            def __init__(self, sbdata, readerfunc, item_size):
                self.sbdata = sbdata
                self.readerfunc = readerfunc
                self.item_size = item_size
            def __getitem__(self,key):
                if isinstance(key,slice):
                    list = []
                    for x in range(*key.indices(self.__len__())):
                        list.append(self.__getitem__(x))
                    return list
                if not (isinstance(key, int)):
                    raise TypeError('must be int')
                key = key * self.item_size # SBData uses byte-based indexes, but we want to use itemsize-based indexes here
                error = SBError()
                my_data = self.readerfunc(self.sbdata,error,key)
                if error.Fail():
                    raise IndexError(error.GetCString())
                else:
                    return my_data
            def __len__(self):
                return int(self.sbdata.GetByteSize()/self.item_size)
            def all(self):
                return self[0:len(self)]

        @classmethod
        def CreateDataFromInt (cls, value, size = None, target = None, ptr_size = None, endian = None):
            import sys
            lldbmodule = sys.modules[cls.__module__]
            lldbdict = lldbmodule.__dict__
            if 'target' in lldbdict:
                lldbtarget = lldbdict['target']
            else:
                lldbtarget = None
            if target == None and lldbtarget != None and lldbtarget.IsValid():
                target = lldbtarget
            if ptr_size == None:
                if target and target.IsValid():
                    ptr_size = target.addr_size
                else:
                    ptr_size = 8
            if endian == None:
                if target and target.IsValid():
                    endian = target.byte_order
                else:
                    endian = lldbdict['eByteOrderLittle']
            if size == None:
                if value > 2147483647:
                    size = 8
                elif value < -2147483648:
                    size = 8
                elif value > 4294967295:
                    size = 8
                else:
                    size = 4
            if size == 4:
                if value < 0:
                    return SBData().CreateDataFromSInt32Array(endian, ptr_size, [value])
                return SBData().CreateDataFromUInt32Array(endian, ptr_size, [value])
            if size == 8:
                if value < 0:
                    return SBData().CreateDataFromSInt64Array(endian, ptr_size, [value])
                return SBData().CreateDataFromUInt64Array(endian, ptr_size, [value])
            return None

        def _make_helper(self, sbdata, getfunc, itemsize):
            return self.read_data_helper(sbdata, getfunc, itemsize)

        def _make_helper_uint8(self):
            return self._make_helper(self, SBData.GetUnsignedInt8, 1)

        def _make_helper_uint16(self):
            return self._make_helper(self, SBData.GetUnsignedInt16, 2)

        def _make_helper_uint32(self):
            return self._make_helper(self, SBData.GetUnsignedInt32, 4)

        def _make_helper_uint64(self):
            return self._make_helper(self, SBData.GetUnsignedInt64, 8)

        def _make_helper_sint8(self):
            return self._make_helper(self, SBData.GetSignedInt8, 1)

        def _make_helper_sint16(self):
            return self._make_helper(self, SBData.GetSignedInt16, 2)

        def _make_helper_sint32(self):
            return self._make_helper(self, SBData.GetSignedInt32, 4)

        def _make_helper_sint64(self):
            return self._make_helper(self, SBData.GetSignedInt64, 8)

        def _make_helper_float(self):
            return self._make_helper(self, SBData.GetFloat, 4)

        def _make_helper_double(self):
            return self._make_helper(self, SBData.GetDouble, 8)

        def _read_all_uint8(self):
            return self._make_helper_uint8().all()

        def _read_all_uint16(self):
            return self._make_helper_uint16().all()

        def _read_all_uint32(self):
            return self._make_helper_uint32().all()

        def _read_all_uint64(self):
            return self._make_helper_uint64().all()

        def _read_all_sint8(self):
            return self._make_helper_sint8().all()

        def _read_all_sint16(self):
            return self._make_helper_sint16().all()

        def _read_all_sint32(self):
            return self._make_helper_sint32().all()

        def _read_all_sint64(self):
            return self._make_helper_sint64().all()

        def _read_all_float(self):
            return self._make_helper_float().all()

        def _read_all_double(self):
            return self._make_helper_double().all()

        uint8 = property(_make_helper_uint8, None, doc='''A read only property that returns an array-like object out of which you can read uint8 values.''')
        uint16 = property(_make_helper_uint16, None, doc='''A read only property that returns an array-like object out of which you can read uint16 values.''')
        uint32 = property(_make_helper_uint32, None, doc='''A read only property that returns an array-like object out of which you can read uint32 values.''')
        uint64 = property(_make_helper_uint64, None, doc='''A read only property that returns an array-like object out of which you can read uint64 values.''')
        sint8 = property(_make_helper_sint8, None, doc='''A read only property that returns an array-like object out of which you can read sint8 values.''')
        sint16 = property(_make_helper_sint16, None, doc='''A read only property that returns an array-like object out of which you can read sint16 values.''')
        sint32 = property(_make_helper_sint32, None, doc='''A read only property that returns an array-like object out of which you can read sint32 values.''')
        sint64 = property(_make_helper_sint64, None, doc='''A read only property that returns an array-like object out of which you can read sint64 values.''')
        float = property(_make_helper_float, None, doc='''A read only property that returns an array-like object out of which you can read float values.''')
        double = property(_make_helper_double, None, doc='''A read only property that returns an array-like object out of which you can read double values.''')
        uint8s = property(_read_all_uint8, None, doc='''A read only property that returns an array with all the contents of this SBData represented as uint8 values.''')
        uint16s = property(_read_all_uint16, None, doc='''A read only property that returns an array with all the contents of this SBData represented as uint16 values.''')
        uint32s = property(_read_all_uint32, None, doc='''A read only property that returns an array with all the contents of this SBData represented as uint32 values.''')
        uint64s = property(_read_all_uint64, None, doc='''A read only property that returns an array with all the contents of this SBData represented as uint64 values.''')
        sint8s = property(_read_all_sint8, None, doc='''A read only property that returns an array with all the contents of this SBData represented as sint8 values.''')
        sint16s = property(_read_all_sint16, None, doc='''A read only property that returns an array with all the contents of this SBData represented as sint16 values.''')
        sint32s = property(_read_all_sint32, None, doc='''A read only property that returns an array with all the contents of this SBData represented as sint32 values.''')
        sint64s = property(_read_all_sint64, None, doc='''A read only property that returns an array with all the contents of this SBData represented as sint64 values.''')
        floats = property(_read_all_float, None, doc='''A read only property that returns an array with all the contents of this SBData represented as float values.''')
        doubles = property(_read_all_double, None, doc='''A read only property that returns an array with all the contents of this SBData represented as double values.''')
        byte_order = property(GetByteOrder, SetByteOrder, doc='''A read/write property getting and setting the endianness of this SBData (data.byte_order = lldb.eByteOrderLittle).''')
        size = property(GetByteSize, None, doc='''A read only property that returns the size the same result as GetByteSize().''')
    %}
#endif
}
