from abc import ABCMeta, abstractmethod
import six

import lldb

@six.add_metaclass(ABCMeta)
class ScriptedProcess:

    """
    The base class for a scripted process.

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.

    DISCLAIMER: THIS INTERFACE IS STILL UNDER DEVELOPMENT AND NOT STABLE.
                THE METHODS EXPOSED MIGHT CHANGE IN THE FUTURE.
    """

    memory_regions = None
    stack_memory_dump = None
    loaded_images = None

    @abstractmethod
    def __init__(self, target, args):
        """ Construct a scripted process.

        Args:
            target (lldb.SBTarget): The target launching the scripted process.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted process.
        """
        self.target = None
        self.args = None
        if isinstance(target, lldb.SBTarget) and target.IsValid():
            self.target = target
        if isinstance(args, lldb.SBStructuredData) and args.IsValid():
            self.args = args

    @abstractmethod
    def get_memory_region_containing_address(self, addr):
        """ Get the memory region for the scripted process, containing a
            specific address.

        Args:
            addr (int): Address to look for in the scripted process memory
                regions.

        Returns:
            lldb.SBMemoryRegionInfo: The memory region containing the address.
                None if out of bounds.
        """
        pass

    @abstractmethod
    def get_thread_with_id(self, tid):
        """ Get the scripted process thread with a specific ID.

        Args:
            tid (int): Thread ID to look for in the scripted process.

        Returns:
            Dict: The thread represented as a dictionary, withr the
                tid thread ID. None if tid doesn't match any of the scripted
                process threads.
        """
        pass

    @abstractmethod
    def get_registers_for_thread(self, tid):
        """ Get the register context dictionary for a certain thread of
            the scripted process.

        Args:
            tid (int): Thread ID for the thread's register context.

        Returns:
            Dict: The register context represented as a dictionary, for the
                tid thread. None if tid doesn't match any of the scripted
                process threads.
        """
        pass

    @abstractmethod
    def read_memory_at_address(self, addr, size):
        """ Get a memory buffer from the scripted process at a certain address,
            of a certain size.

        Args:
            addr (int): Address from which we should start reading.
            size (int): Size of the memory to read.

        Returns:
            lldb.SBData: An `lldb.SBData` buffer with the target byte size and
                byte order storing the memory read.
        """
        pass

    def get_loaded_images(self):
        """ Get the list of loaded images for the scripted process.

        ```
        class ScriptedProcessImage:
            def __init__(name, file_spec, uuid, load_address):
              self.name = name
              self.file_spec = file_spec
              self.uuid = uuid
              self.load_address = load_address
        ```

        Returns:
            List[ScriptedProcessImage]: A list of `ScriptedProcessImage`
                containing for each entry, the name of the library, a UUID,
                an `lldb.SBFileSpec` and a load address.
                None if the list is empty.
        """
        return self.loaded_images

    def get_process_id(self):
        """ Get the scripted process identifier.

        Returns:
            int: The scripted process identifier.
        """
        return 0


    def launch(self):
        """ Simulate the scripted process launch.

        Returns:
            lldb.SBError: An `lldb.SBError` with error code 0.
        """
        return lldb.SBError()

    def resume(self):
        """ Simulate the scripted process resume.

        Returns:
            lldb.SBError: An `lldb.SBError` with error code 0.
        """
        return lldb.SBError()

    @abstractmethod
    def should_stop(self):
        """ Check if the scripted process plugin should produce the stop event.

        Returns:
            bool: True if scripted process should broadcast a stop event.
                  False otherwise.
        """
        pass

    def stop(self):
        """ Trigger the scripted process stop.

        Returns:
            lldb.SBError: An `lldb.SBError` with error code 0.
        """
        return lldb.SBError()

    @abstractmethod
    def is_alive(self):
        """ Check if the scripted process is alive.

        Returns:
            bool: True if scripted process is alive. False otherwise.
        """
        pass

    @abstractmethod
    def get_scripted_thread_plugin(self):
        """ Get scripted thread plugin name.

        Returns:
            str: Name of the scripted thread plugin.
        """
        return None

@six.add_metaclass(ABCMeta)
class ScriptedThread:

    """
    The base class for a scripted thread.

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.

    DISCLAIMER: THIS INTERFACE IS STILL UNDER DEVELOPMENT AND NOT STABLE.
                THE METHODS EXPOSED MIGHT CHANGE IN THE FUTURE.
    """

    @abstractmethod
    def __init__(self, process, args):
        """ Construct a scripted thread.

        Args:
            process (lldb.SBProcess): The scripted process owning this thread.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted thread.
        """
        self.target = None
        self.process = None
        self.args = None
        if isinstance(process, lldb.SBProcess) and process.IsValid():
            self.process = process
            self.target = process.GetTarget()

        self.id = None
        self.name = None
        self.queue = None
        self.state = None
        self.stop_reason = None
        self.register_info = None
        self.register_ctx = {}
        self.frames = []

    @abstractmethod
    def get_thread_id(self):
        """ Get the scripted thread identifier.

        Returns:
            int: The identifier of the scripted thread.
        """
        pass

    @abstractmethod
    def get_name(self):
        """ Get the scripted thread name.

        Returns:
            str: The name of the scripted thread.
        """
        pass

    def get_state(self):
        """ Get the scripted thread state type.

            eStateStopped,   ///< Process or thread is stopped and can be examined.
            eStateRunning,   ///< Process or thread is running and can't be examined.
            eStateStepping,  ///< Process or thread is in the process of stepping and can
                             /// not be examined.

        Returns:
            int: The state type of the scripted thread.
                 Returns lldb.eStateStopped by default.
        """
        return lldb.eStateStopped

    def get_queue(self):
        """ Get the scripted thread associated queue name.
            This method is optional.

        Returns:
            str: The queue name associated with the scripted thread.
        """
        pass

    @abstractmethod
    def get_stop_reason(self):
        """ Get the dictionary describing the stop reason type with some data.
            This method is optional.

        Returns:
            Dict: The dictionary holding the stop reason type and the possibly
            the stop reason data.
        """
        pass

    def get_stackframes(self):
        """ Get the list of stack frames for the scripted thread.

        ```
        class ScriptedStackFrame:
            def __init__(idx, cfa, pc, symbol_ctx):
                self.idx = idx
                self.cfa = cfa
                self.pc = pc
                self.symbol_ctx = symbol_ctx
        ```

        Returns:
            List[ScriptedFrame]: A list of `ScriptedStackFrame`
                containing for each entry, the frame index, the canonical
                frame address, the program counter value for that frame
                and a symbol context.
                None if the list is empty.
        """
        return 0

    def get_register_info(self):
        if self.register_info is None:
            self.register_info = dict()
            triple = self.target.triple
            if triple:
                arch = triple.split('-')[0]
                if arch == 'x86_64':
                    self.register_info['sets'] = ['General Purpose Registers']
                    self.register_info['registers'] = [
                        {'name': 'rax', 'bitsize': 64, 'offset': 0, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 0, 'dwarf': 0},
                        {'name': 'rbx', 'bitsize': 64, 'offset': 8, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 3, 'dwarf': 3},
                        {'name': 'rcx', 'bitsize': 64, 'offset': 16, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 2, 'dwarf': 2, 'generic': 'arg4', 'alt-name': 'arg4'},
                        {'name': 'rdx', 'bitsize': 64, 'offset': 24, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 1, 'dwarf': 1, 'generic': 'arg3', 'alt-name': 'arg3'},
                        {'name': 'rdi', 'bitsize': 64, 'offset': 32, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 5, 'dwarf': 5, 'generic': 'arg1', 'alt-name': 'arg1'},
                        {'name': 'rsi', 'bitsize': 64, 'offset': 40, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 4, 'dwarf': 4, 'generic': 'arg2', 'alt-name': 'arg2'},
                        {'name': 'rbp', 'bitsize': 64, 'offset': 48, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 6, 'dwarf': 6, 'generic': 'fp', 'alt-name': 'fp'},
                        {'name': 'rsp', 'bitsize': 64, 'offset': 56, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 7, 'dwarf': 7, 'generic': 'sp', 'alt-name': 'sp'},
                        {'name': 'r8', 'bitsize': 64, 'offset': 64, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 8, 'dwarf': 8, 'generic': 'arg5', 'alt-name': 'arg5'},
                        {'name': 'r9', 'bitsize': 64, 'offset': 72, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 9, 'dwarf': 9, 'generic': 'arg6', 'alt-name': 'arg6'},
                        {'name': 'r10', 'bitsize': 64, 'offset': 80, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 10, 'dwarf': 10},
                        {'name': 'r11', 'bitsize': 64, 'offset': 88, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 11, 'dwarf': 11},
                        {'name': 'r12', 'bitsize': 64, 'offset': 96, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 12, 'dwarf': 12},
                        {'name': 'r13', 'bitsize': 64, 'offset': 104, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 13, 'dwarf': 13},
                        {'name': 'r14', 'bitsize': 64, 'offset': 112, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 14, 'dwarf': 14},
                        {'name': 'r15', 'bitsize': 64, 'offset': 120, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 15, 'dwarf': 15},
                        {'name': 'rip', 'bitsize': 64, 'offset': 128, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 16, 'dwarf': 16, 'generic': 'pc', 'alt-name': 'pc'},
                        {'name': 'rflags', 'bitsize': 64, 'offset': 136, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'generic': 'flags', 'alt-name': 'flags'},
                        {'name': 'cs', 'bitsize': 64, 'offset': 144, 'encoding': 'uint', 'format': 'hex', 'set': 0},
                        {'name': 'fs', 'bitsize': 64, 'offset': 152, 'encoding': 'uint', 'format': 'hex', 'set': 0},
                        {'name': 'gs', 'bitsize': 64, 'offset': 160, 'encoding': 'uint', 'format': 'hex', 'set': 0}
                    ]
                elif 'arm64' in arch:
                    self.register_info['sets'] = ['General Purpose Registers']
                    self.register_info['registers'] = [
                        {'name': 'x0',   'bitsize': 64, 'offset': 0,   'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 0,  'dwarf': 0,  'generic': 'arg0', 'alt-name': 'arg0'},
                        {'name': 'x1',   'bitsize': 64, 'offset': 8,   'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 1,  'dwarf': 1,  'generic': 'arg1', 'alt-name': 'arg1'},
                        {'name': 'x2',   'bitsize': 64, 'offset': 16,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 2,  'dwarf': 2,  'generic': 'arg2', 'alt-name': 'arg2'},
                        {'name': 'x3',   'bitsize': 64, 'offset': 24,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 3,  'dwarf': 3,  'generic': 'arg3', 'alt-name': 'arg3'},
                        {'name': 'x4',   'bitsize': 64, 'offset': 32,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 4,  'dwarf': 4,  'generic': 'arg4', 'alt-name': 'arg4'},
                        {'name': 'x5',   'bitsize': 64, 'offset': 40,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 5,  'dwarf': 5,  'generic': 'arg5', 'alt-name': 'arg5'},
                        {'name': 'x6',   'bitsize': 64, 'offset': 48,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 6,  'dwarf': 6,  'generic': 'arg6', 'alt-name': 'arg6'},
                        {'name': 'x7',   'bitsize': 64, 'offset': 56,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 7,  'dwarf': 7,  'generic': 'arg7', 'alt-name': 'arg7'},
                        {'name': 'x8',   'bitsize': 64, 'offset': 64,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 8,  'dwarf': 8 },
                        {'name': 'x9',   'bitsize': 64, 'offset': 72,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 9,  'dwarf': 9 },
                        {'name': 'x10',  'bitsize': 64, 'offset': 80,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 10, 'dwarf': 10},
                        {'name': 'x11',  'bitsize': 64, 'offset': 88,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 11, 'dwarf': 11},
                        {'name': 'x12',  'bitsize': 64, 'offset': 96,  'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 12, 'dwarf': 12},
                        {'name': 'x13',  'bitsize': 64, 'offset': 104, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 13, 'dwarf': 13},
                        {'name': 'x14',  'bitsize': 64, 'offset': 112, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 14, 'dwarf': 14},
                        {'name': 'x15',  'bitsize': 64, 'offset': 120, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 15, 'dwarf': 15},
                        {'name': 'x16',  'bitsize': 64, 'offset': 128, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 16, 'dwarf': 16},
                        {'name': 'x17',  'bitsize': 64, 'offset': 136, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 17, 'dwarf': 17},
                        {'name': 'x18',  'bitsize': 64, 'offset': 144, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 18, 'dwarf': 18},
                        {'name': 'x19',  'bitsize': 64, 'offset': 152, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 19, 'dwarf': 19},
                        {'name': 'x20',  'bitsize': 64, 'offset': 160, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 20, 'dwarf': 20},
                        {'name': 'x21',  'bitsize': 64, 'offset': 168, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 21, 'dwarf': 21},
                        {'name': 'x22',  'bitsize': 64, 'offset': 176, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 22, 'dwarf': 22},
                        {'name': 'x23',  'bitsize': 64, 'offset': 184, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 23, 'dwarf': 23},
                        {'name': 'x24',  'bitsize': 64, 'offset': 192, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 24, 'dwarf': 24},
                        {'name': 'x25',  'bitsize': 64, 'offset': 200, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 25, 'dwarf': 25},
                        {'name': 'x26',  'bitsize': 64, 'offset': 208, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 26, 'dwarf': 26},
                        {'name': 'x27',  'bitsize': 64, 'offset': 216, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 27, 'dwarf': 27},
                        {'name': 'x28',  'bitsize': 64, 'offset': 224, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 28, 'dwarf': 28},
                        {'name': 'x29',  'bitsize': 64, 'offset': 232, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 29, 'dwarf': 29, 'generic': 'fp', 'alt-name': 'fp'},
                        {'name': 'x30',  'bitsize': 64, 'offset': 240, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 30, 'dwarf': 30, 'generic': 'lr', 'alt-name': 'lr'},
                        {'name': 'sp',   'bitsize': 64, 'offset': 248, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 31, 'dwarf': 31, 'generic': 'sp', 'alt-name': 'sp'},
                        {'name': 'pc',   'bitsize': 64, 'offset': 256, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 32, 'dwarf': 32, 'generic': 'pc', 'alt-name': 'pc'},
                        {'name': 'cpsr', 'bitsize': 32, 'offset': 264, 'encoding': 'uint', 'format': 'hex', 'set': 0, 'gcc': 33, 'dwarf': 33}
                    ]
                else: raise ValueError('Unknown architecture', arch)
        return self.register_info

    @abstractmethod
    def get_register_context(self):
        """ Get the scripted thread register context

        Returns:
            str: A byte representing all register's value.
        """
        pass
