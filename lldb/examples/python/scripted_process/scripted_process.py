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
    def get_memory_region_containing_address(addr):
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
    def get_thread_with_id(tid):
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
    def get_registers_for_thread(tid):
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
    def read_memory_at_address(addr, size):
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

    @abstractmethod
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
        pass

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
