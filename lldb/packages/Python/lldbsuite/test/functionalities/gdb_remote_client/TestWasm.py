import lldb
import binascii
import struct
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *

LLDB_INVALID_ADDRESS = 0xffffffffffffffff
load_address = 0x400000000

def format_register_value(val):
    """
    Encode each byte by two hex digits in little-endian order.
    """
    return ''.join(x.encode('hex') for x in struct.pack('<Q', val))

def uleb128_encode(val):
    """
    encode a number to uleb128
    """
    result = bytearray()
    while True:
        byte = val & 0x7f
        val >>= 7
        if val != 0:
            byte |= 0x80  # mark this byte to show that more bytes will follow
        result.append(byte)
        if val == 0:
            break
    return result


def encode_wasm_string(s):
    """
    Encode a string as an array of UTF-8 bytes preceded by its ULEB128 length.
    """
    char_array = bytearray(x.encode("utf8") for x in s)
    return uleb128_encode(len(char_array)) + char_array


def format_bytearray_as_hexstring(byte_array):
    """
    Encode a n array of bytes as a string of hexadecimal digits.
    """
    return ''.join(format(x, '02x') for x in byte_array)


class MyResponder(MockGDBServerResponder):
    current_pc = load_address + 0x0a

    def __init__(self, obj_path):
        self._obj_path = obj_path
        MockGDBServerResponder.__init__(self)

    def respond(self, packet):
        if packet == "qProcessInfo":
            return self.qProcessInfo()
        if packet[0:13] == "qRegisterInfo":
            return self.qRegisterInfo(packet[13:])
        return MockGDBServerResponder.respond(self, packet)

    def qSupported(self, client_supported):
        return "qXfer:libraries:read+;PacketSize=1000;vContSupported-"

    def qHostInfo(self):
        return ""

    def QEnableErrorStrings(self):
        return ""

    def qfThreadInfo(self):
        return "OK"

    def qRegisterInfo(self, index):
        if int(index) == 0:
            return "name:pc;alt-name:pc;bitsize:64;offset:0;encoding:uint;format:hex;set:General Purpose Registers;gcc:16;dwarf:16;generic:pc;"
        return "E45"

    def qProcessInfo(self):
        return "pid:1;ppid:1;uid:1;gid:1;euid:1;egid:1;name:%s;triple:%s;ptrsize:4" % (hex_encode_bytes("lldb"), hex_encode_bytes("wasm32-unknown-unknown-wasm"))

    def haltReason(self):
        return "T05thread-pcs:" + format(self.current_pc, 'x') + ";thread:1;"

    def readRegister(self, register):
        return format_register_value(self.current_pc)

    def qXferRead(self, obj, annex, offset, length):
        if obj == "libraries":
            xml = '<library-list><library name=\"%s\"><section address=\"%d\"/></library></library-list>' % ("test_wasm", load_address)
            return xml, False
        else:
            return None, False

    def readMemory(self, addr, length):
        if addr < load_address:
            return "E02"
        result = ""
        with open(self._obj_path, mode='rb') as file:
            file_content = bytearray(file.read())
            addr_from = addr - load_address
            addr_to = addr_from + min(length, len(file_content) - addr_from)
            for i in range(addr_from, addr_to):
                result += format(file_content[i], '02x')
            file.close()
        return result


class TestWasm(GDBRemoteTestBase):

    def setUp(self):
        super(TestWasm, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(TestWasm, self).tearDown()

    def test_load_module_with_embedded_symbols_from_remote(self):
        """Test connecting to a WebAssembly engine via GDB-remote and loading a Wasm module with embedded DWARF symbols"""

        yaml_path = "test_wasm_embedded_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder(obj_path)

        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateStopped])

        num_modules = target.GetNumModules()
        self.assertEquals(1, num_modules)

        module = target.GetModuleAtIndex(0)
        num_sections = module.GetNumSections()
        self.assertEquals(5, num_sections)

        code_section = module.GetSectionAtIndex(0)
        self.assertEquals("code", code_section.GetName())
        self.assertEquals(load_address | code_section.GetFileOffset(), code_section.GetLoadAddress(target))

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEquals(".debug_info", debug_info_section.GetName())
        self.assertEquals(load_address | debug_info_section.GetFileOffset(), debug_info_section.GetLoadAddress(target))

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEquals(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEquals(load_address | debug_abbrev_section.GetFileOffset(), debug_abbrev_section.GetLoadAddress(target))

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEquals(".debug_line", debug_line_section.GetName())
        self.assertEquals(load_address | debug_line_section.GetFileOffset(), debug_line_section.GetLoadAddress(target))

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEquals(".debug_str", debug_str_section.GetName())
        self.assertEquals(load_address | debug_line_section.GetFileOffset(), debug_line_section.GetLoadAddress(target))


    def test_load_module_with_stripped_symbols_from_remote(self):
        """Test connecting to a WebAssembly engine via GDB-remote and loading a Wasm module with symbols stripped into a separate Wasm file"""

        sym_yaml_path = "test_sym.yaml"
        sym_yaml_base, ext = os.path.splitext(sym_yaml_path)
        sym_obj_path = self.getBuildArtifact(sym_yaml_base) + ".wasm"
        self.yaml2obj(sym_yaml_path, sym_obj_path)

        yaml_template_path = "test_wasm_external_debug_sections.yaml"
        yaml_base = "test_wasm_external_debug_sections_modified"
        yaml_path = self.getBuildArtifact(yaml_base) + ".yaml"
        obj_path = self.getBuildArtifact(yaml_base) + ".wasm"
        with open(yaml_template_path, mode='r') as file:
            yaml = file.read()
            file.close()
            yaml = yaml.replace("###_EXTERNAL_DEBUG_INFO_###", format_bytearray_as_hexstring(encode_wasm_string(sym_obj_path)))
            with open(yaml_path, mode='w') as file:
                file.write(yaml)
                file.close()
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = MyResponder(obj_path)

        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateStopped])
    
        num_modules = target.GetNumModules()
        self.assertEquals(1, num_modules)

        module = target.GetModuleAtIndex(0)
        num_sections = module.GetNumSections()
        self.assertEquals(5, num_sections)

        code_section = module.GetSectionAtIndex(0)
        self.assertEquals("code", code_section.GetName())
        self.assertEquals(load_address | code_section.GetFileOffset(), code_section.GetLoadAddress(target))

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEquals(".debug_info", debug_info_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_info_section.GetLoadAddress(target))

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEquals(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_abbrev_section.GetLoadAddress(target))

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEquals(".debug_line", debug_line_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target))

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEquals(".debug_str", debug_str_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target))


    def test_load_module_from_file(self):
        """Test connecting to a WebAssembly engine via GDB-remote and loading a Wasm module from a file"""

        class Responder(MyResponder):

            def __init__(self, obj_path):
                MyResponder.__init__(self, obj_path)

            def qXferRead(self, obj, annex, offset, length):
                if obj == "libraries":
                    xml = '<library-list><library name=\"%s\"><section address=\"%d\"/></library></library-list>' % (self._obj_path, load_address)
                    print xml
                    return xml, False
                else:
                    return None, False

            def readMemory(self, addr, length):
                assert False # Should not be called


        yaml_path = "test_wasm_embedded_debug_sections.yaml"
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        self.server.responder = Responder(obj_path)

        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateStopped])
    
        num_modules = target.GetNumModules()
        self.assertEquals(1, num_modules)

        module = target.GetModuleAtIndex(0)
        num_sections = module.GetNumSections()
        self.assertEquals(5, num_sections)

        code_section = module.GetSectionAtIndex(0)
        self.assertEquals("code", code_section.GetName())
        self.assertEquals(load_address | code_section.GetFileOffset(), code_section.GetLoadAddress(target))

        debug_info_section = module.GetSectionAtIndex(1)
        self.assertEquals(".debug_info", debug_info_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_info_section.GetLoadAddress(target))

        debug_abbrev_section = module.GetSectionAtIndex(2)
        self.assertEquals(".debug_abbrev", debug_abbrev_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_abbrev_section.GetLoadAddress(target))

        debug_line_section = module.GetSectionAtIndex(3)
        self.assertEquals(".debug_line", debug_line_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target))

        debug_str_section = module.GetSectionAtIndex(4)
        self.assertEquals(".debug_str", debug_str_section.GetName())
        self.assertEquals(LLDB_INVALID_ADDRESS, debug_line_section.GetLoadAddress(target))

