"""
Python LLDB data formatter for libc++ std::vector

1-to-1 translation from the LLDB builtin std::vector formatter.

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import lldb


def get_data_pointer(root):
    """Get the data pointer from a vector, handling compressed pair layout."""
    # Try new layout
    cap_sp = root.GetChildMemberWithName("__cap_")
    if cap_sp:
        return cap_sp

    # Try old compressed pair layout
    end_cap_sp = root.GetChildMemberWithName("__end_cap_")
    if not end_cap_sp:
        return None

    # Get first value of compressed pair
    value_sp = end_cap_sp.GetChildMemberWithName("__value_")
    if value_sp:
        return value_sp

    first_sp = end_cap_sp.GetChildMemberWithName("__first_")
    if first_sp:
        return first_sp

    return None


class LibCxxStdVectorSyntheticFrontEnd:
    """Synthetic children frontend for libc++ std::vector."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.m_start = None
        self.m_finish = None
        self.m_element_type = None
        self.m_element_size = 0

    def num_children(self):
        if not self.m_start or not self.m_finish:
            return 0

        start_val = self.m_start.GetValueAsUnsigned(0)
        finish_val = self.m_finish.GetValueAsUnsigned(0)

        if start_val == 0 or finish_val == 0:
            return 0

        if start_val > finish_val:
            return 0

        num_children = finish_val - start_val
        if num_children % self.m_element_size != 0:
            return 0

        return num_children // self.m_element_size

    def get_child_at_index(self, index):
        if not self.m_start or not self.m_finish:
            return None

        offset = index * self.m_element_size
        offset = offset + self.m_start.GetValueAsUnsigned(0)

        name = "[%d]" % index
        target = self.valobj.GetTarget()
        if not target:
            return None

        addr = lldb.SBAddress(offset, target)
        return target.CreateValueFromAddress(name, addr, self.m_element_type)

    def update(self):
        self.m_start = None
        self.m_finish = None

        data_sp = get_data_pointer(self.valobj)
        if not data_sp:
            return False

        self.m_element_type = data_sp.GetType().GetPointeeType()
        if not self.m_element_type:
            return False

        size = self.m_element_type.GetByteSize()
        if not size or size == 0:
            return False

        self.m_element_size = size

        begin_sp = self.valobj.GetChildMemberWithName("__begin_")
        end_sp = self.valobj.GetChildMemberWithName("__end_")

        if not begin_sp:
            return False

        self.m_start = begin_sp
        self.m_finish = end_sp

        return True


class LibCxxVectorBoolSyntheticFrontEnd:
    """Synthetic children frontend for libc++ std::vector<bool>."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.m_bool_type = None
        self.m_count = 0
        self.m_base_data_address = 0
        self.m_children = {}

        # Get bool type
        if valobj:
            target = valobj.GetTarget()
            if target:
                self.m_bool_type = target.GetBasicType(lldb.eBasicTypeBool)

    def num_children(self):
        return self.m_count

    def get_child_at_index(self, index):
        if index in self.m_children:
            return self.m_children[index]

        if index >= self.m_count:
            return None

        if self.m_base_data_address == 0 or self.m_count == 0:
            return None

        if not self.m_bool_type:
            return None

        # Calculate byte and bit index
        byte_idx = index >> 3  # divide by 8
        bit_index = index & 7  # modulo 8

        byte_location = self.m_base_data_address + byte_idx

        process = self.valobj.GetProcess()
        if not process:
            return None

        error = lldb.SBError()
        byte_data = process.ReadMemory(byte_location, 1, error)
        if error.Fail() or not byte_data or len(byte_data) == 0:
            return None

        byte = ord(byte_data[0:1])
        mask = 1 << bit_index
        bit_set = (byte & mask) != 0

        bool_size = self.m_bool_type.GetByteSize()
        if not bool_size:
            return None

        # Create data for the bool value
        data = lldb.SBData()
        data.SetData(
            error,
            bytes([1 if bit_set else 0]),
            process.GetByteOrder(),
            process.GetAddressByteSize(),
        )

        name = "[%d]" % index
        target = self.valobj.GetTarget()
        if not target:
            return None

        retval_sp = target.CreateValueFromData(name, data, self.m_bool_type)
        if retval_sp:
            self.m_children[index] = retval_sp

        return retval_sp

    def update(self):
        self.m_children = {}

        if not self.valobj:
            return False

        size_sp = self.valobj.GetChildMemberWithName("__size_")
        if not size_sp:
            return False

        self.m_count = size_sp.GetValueAsUnsigned(0)
        if not self.m_count:
            return True

        begin_sp = self.valobj.GetChildMemberWithName("__begin_")
        if not begin_sp:
            self.m_count = 0
            return False

        self.m_base_data_address = begin_sp.GetValueAsUnsigned(0)
        if not self.m_base_data_address:
            self.m_count = 0
            return False

        return True


def LibCxxStdVectorSyntheticFrontendCreator(valobj, internal_dict):
    if not valobj:
        return None

    compiler_type = valobj.GetType()
    if not compiler_type or compiler_type.GetNumberOfTemplateArguments() == 0:
        return None

    arg_type = compiler_type.GetTemplateArgumentType(0)
    if arg_type.GetName() == "bool":
        return LibCxxVectorBoolSyntheticFrontEnd(valobj, internal_dict)

    return LibCxxStdVectorSyntheticFrontEnd(valobj, internal_dict)
