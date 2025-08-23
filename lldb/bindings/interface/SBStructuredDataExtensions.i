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
            return lldb_iter(self, 'GetSize', 'GetItemAtIndex')
        elif data_type == eStructuredDataTypeDictionary:
            keys = SBStringList()
            self.GetKeys(keys)
            return iter(keys)
        raise TypeError(f"cannot iterate {self.type_name(data_type)} type")

    def __getitem__(self, key):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeArray:
            if not isinstance(key, int):
                raise TypeError("subscript index must be an integer")
            count = len(self)
            if -count <= key < count:
                key %= count
                return self.GetItemAtIndex(key)
            raise IndexError("index out of range")
        elif data_type == eStructuredDataTypeDictionary:
            if not isinstance(key, str):
                raise TypeError("subscript key must be a string")
            return self.GetValueForKey(key)
        else:
            raise TypeError(f"cannot subscript {self.type_name(data_type)} type")

    def __bool__(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeBoolean:
            return self.GetBooleanValue()
        elif data_type == eStructuredDataTypeInteger:
            return bool(int(self))
        elif data_type == eStructuredDataTypeSignedInteger:
            return bool(int(self))
        elif data_type == eStructuredDataTypeFloat:
            return bool(float(self))
        elif data_type == eStructuredDataTypeString:
            return bool(str(self))
        elif data_type == eStructuredDataTypeArray:
            return bool(len(self))
        elif data_type == eStructuredDataTypeDictionary:
            return bool(len(self))
        elif data_type == eStructuredDataTypeNull:
            return False
        elif data_type == eStructuredDataTypeInvalid:
            return False
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to bool")

    def __str__(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeString:
            size = len(self) or 1023
            return self.GetStringValue(size + 1)
        elif data_type == eStructuredDataTypeInteger:
            return str(int(self))
        elif data_type == eStructuredDataTypeSignedInteger:
            return str(int(self))
        elif data_type == eStructuredDataTypeFloat:
            return str(float(self))
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to string")

    def __int__(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeInteger:
            return self.GetUnsignedIntegerValue()
        elif data_type == eStructuredDataTypeSignedInteger:
            return self.GetSignedIntegerValue()
        elif data_type == eStructuredDataTypeFloat:
            return int(float(self))
        elif data_type == eStructuredDataTypeString:
            return int(str(self))
        elif data_type == eStructuredDataTypeBoolean:
            return int(bool(self))
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to int")

    def __float__(self):
        data_type = self.GetType()
        if data_type == eStructuredDataTypeFloat:
            return self.GetFloatValue()
        elif data_type == eStructuredDataTypeInteger:
            return float(int(self))
        elif data_type == eStructuredDataTypeSignedInteger:
            return float(int(self))
        elif data_type == eStructuredDataTypeString:
            return float(str(self))
        else:
            raise TypeError(f"cannot convert {self.type_name(data_type)} to float")

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
