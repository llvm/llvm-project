# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# LLDB data formatter for libc++ std::vector


import lldb
from lldb.plugins.scripted_synthetic_children import ScriptedSyntheticChildren


class LibCxxStdVectorSyntheticFrontEnd(ScriptedSyntheticChildren):
    """Synthetic children frontend for libc++ std::vector."""

    def __init__(self, valobj, internal_dict):
        super().__init__(valobj)
        self.m_start = None
        # Either the `__end_` pointer or the `__size_` integer, depending on
        # which layout libc++ was built with. See `m_size_based`.
        self.m_finish = None
        self.m_size_based = False
        self.m_element_type = None
        self.m_element_size = 0

    def num_children(self):
        if not self.m_start or not self.m_finish:
            return 0

        if self.m_size_based:
            return self.m_finish.GetValueAsUnsigned(0)

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

        name = f"[{index}]"
        # Create the child from the vector's SBValue rather than the SBTarget:
        # LLDB caches synthetic children, and only children created this way
        # are kept alive by the parent once Python drops its reference.
        return self.valobj.CreateValueFromAddress(name, offset, self.m_element_type)

    def update(self):
        self.m_start = None
        self.m_finish = None

        # Newer libc++ versions keep the vector's members in a `__layout_`
        # subobject rather than directly in the vector.
        members = self.valobj.GetChildMemberWithName("__layout_")
        if not members:
            members = self.valobj

        begin_sp = members.GetChildMemberWithName("__begin_")
        if not begin_sp:
            return False

        self.m_element_type = begin_sp.GetType().GetPointeeType()
        if not self.m_element_type:
            return False

        size = self.m_element_type.GetByteSize()
        if not size:
            return False

        self.m_element_size = size
        self.m_start = begin_sp

        end_sp = members.GetChildMemberWithName("__end_")
        if end_sp:
            self.m_finish = end_sp
            self.m_size_based = False
            return False

        size_sp = members.GetChildMemberWithName("__size_")
        if not size_sp:
            return False

        self.m_finish = size_sp
        self.m_size_based = True
        return False


class LibCxxVectorBoolSyntheticFrontEnd(ScriptedSyntheticChildren):
    """Synthetic children frontend for libc++ std::vector<bool>."""

    def __init__(self, valobj, internal_dict):
        super().__init__(valobj)
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
        retval_sp = self.valobj.CreateValueFromData(name, data, self.m_bool_type)
        if retval_sp:
            self.m_children[index] = retval_sp

        return retval_sp

    def update(self):
        """
        Returning False asks LLDB to re-run this on every stop. Returning True
        freezes the children we last reported, so it is only safe when there
        are none to go stale.
        """
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

        return False


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
