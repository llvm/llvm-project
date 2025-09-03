STRING_EXTENSION_OUTSIDE(SBStructuredData)

%extend lldb::SBStructuredData {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of element in a lldb.SBStructuredData object.'''
      return self.GetSize()

    def __iter__(self):
        '''Iterate over all the elements in a lldb.SBStructuredData object.'''
        data_type = self.GetType()
        if data_type == eStructuredDataTypeArray:
            for i in range(self.GetSize()):
                yield self.GetItemAtIndex(i).dynamic
            return
        elif data_type == eStructuredDataTypeDictionary:
            keys = SBStringList()
            self.GetKeys(keys)
            return iter(keys)
        else:
            raise TypeError(f"cannot iterate {self.type_name(data_type)} type")

    def __getitem__(self, key):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeArray:
            if not isinstance(key, int):
                raise TypeError("subscript index must be an integer")
            count = len(self)
            if -count <= key < count:
                key %= count
                return self.GetItemAtIndex(key).dynamic
            raise IndexError("index out of range")
        elif data_type == eStructuredDataTypeDictionary:
            if not isinstance(key, str):
                raise TypeError("subscript key must be a string")
            return self.GetValueForKey(key).dynamic
        else:
            raise TypeError(f"cannot subscript {self.type_name(data_type)} type")

    def __bool__(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeInvalid:
            return False
        elif data_type in (
            eStructuredDataTypeArray,
            eStructuredDataTypeDictionary,
        ):
            return self.GetSize() != 0
        elif data_type != eStructuredDataTypeGeneric:
            return bool(self.dynamic)
        else:
            raise TypeError("cannot convert generic to bool")

    def __int__(self):
        data_type = self.GetType()
        if data_type in (
            eStructuredDataTypeInteger,
            eStructuredDataTypeSignedInteger,
        ):
            return int(self.dynamic)
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to int")

    def __float__(self):
        data_type = self.GetType()
        if data_type in (
            eStructuredDataTypeFloat,
            eStructuredDataTypeInteger,
            eStructuredDataTypeSignedInteger,
        ):
            return float(self.dynamic)
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to float")

    @property
    def dynamic(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeNull:
            return None
        elif data_type == eStructuredDataTypeBoolean:
            return self.GetBooleanValue()
        elif data_type == eStructuredDataTypeInteger:
            return self.GetUnsignedIntegerValue()
        elif data_type == eStructuredDataTypeSignedInteger:
            return self.GetSignedIntegerValue()
        elif data_type == eStructuredDataTypeFloat:
            return self.GetFloatValue()
        elif data_type == eStructuredDataTypeString:
            size = len(self) or 1023
            return self.GetStringValue(size + 1)
        elif data_type == eStructuredDataTypeGeneric:
            return self.GetGenericValue()
        else:
            return self

    @staticmethod
    def type_name(t):
        if t == eStructuredDataTypeNull:
            return "null"
        elif t == eStructuredDataTypeBoolean:
            return "boolean"
        elif t == eStructuredDataTypeInteger:
            return "integer"
        elif t == eStructuredDataTypeSignedInteger:
            return "integer"
        elif t == eStructuredDataTypeFloat:
            return "float"
        elif t == eStructuredDataTypeString:
            return "string"
        elif t == eStructuredDataTypeArray:
            return "array"
        elif t == eStructuredDataTypeDictionary:
            return "dictionary"
        elif t == eStructuredDataTypeGeneric:
            return "generic"
        elif t == eStructuredDataTypeInvalid:
            return "invalid"
        else:
            raise TypeError(f"unknown structured data type: {t}")
    %}
#endif
}
