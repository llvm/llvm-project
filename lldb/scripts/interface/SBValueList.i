//===-- SWIG Interface for SBValueList --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a collection of SBValues.  Both SBFrame's GetVariables() and
GetRegisters() return a SBValueList.

SBValueList supports SBValue iteration. For example (from test/lldbutil.py),

def get_registers(frame, kind):
    '''Returns the registers given the frame and the kind of registers desired.

    Returns None if there's no such kind.
    '''
    registerSet = frame.GetRegisters() # Return type of SBValueList.
    for value in registerSet:
        if kind.lower() in value.GetName().lower():
            return value

    return None

def get_GPRs(frame):
    '''Returns the general purpose registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_GPRs
        regs = get_GPRs(frame)
        for reg in regs:
            print('%s => %s' % (reg.GetName(), reg.GetValue()))
        ...
    '''
    return get_registers(frame, 'general purpose')

def get_FPRs(frame):
    '''Returns the floating point registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_FPRs
        regs = get_FPRs(frame)
        for reg in regs:
            print('%s => %s' % (reg.GetName(), reg.GetValue()))
        ...
    '''
    return get_registers(frame, 'floating point')

def get_ESRs(frame):
    '''Returns the exception state registers of the frame as an SBValue.

    The returned SBValue object is iterable.  An example:
        ...
        from lldbutil import get_ESRs
        regs = get_ESRs(frame)
        for reg in regs:
            print('%s => %s' % (reg.GetName(), reg.GetValue()))
        ...
    '''
    return get_registers(frame, 'exception state')"
) SBValueList;
class SBValueList
{
public:

    SBValueList ();

    SBValueList (const lldb::SBValueList &rhs);

    ~SBValueList();

    bool
    IsValid() const;

    explicit operator bool() const;
    
    void 
    Clear();

    void
    Append (const lldb::SBValue &val_obj);

    void
    Append (const lldb::SBValueList& value_list);

    uint32_t
    GetSize() const;

    lldb::SBValue
    GetValueAtIndex (uint32_t idx) const;

    lldb::SBValue
    FindValueObjectByUID (lldb::user_id_t uid);
    
    lldb::SBValue
    GetFirstValueByName (const char* name) const;
    
    %pythoncode %{
        def __len__(self):
            return int(self.GetSize())

        def __getitem__(self, key):
            count = len(self)
            #------------------------------------------------------------
            # Access with "int" to get Nth item in the list
            #------------------------------------------------------------
            if type(key) is int:
                if key < count:
                    return self.GetValueAtIndex(key)
            #------------------------------------------------------------
            # Access with "str" to get values by name
            #------------------------------------------------------------
            elif type(key) is str:
                matches = []
                for idx in range(count):
                    value = self.GetValueAtIndex(idx)
                    if value.name == key:
                        matches.append(value)
                return matches
            #------------------------------------------------------------
            # Match with regex
            #------------------------------------------------------------
            elif isinstance(key, type(re.compile('.'))):
                matches = []
                for idx in range(count):
                    value = self.GetValueAtIndex(idx)
                    re_match = key.search(value.name)
                    if re_match:
                        matches.append(value)
                return matches

    %}


};

} // namespace lldb
