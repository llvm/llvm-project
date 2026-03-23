from abc import ABCMeta, abstractmethod

import lldb


class ScriptedFrameProvider(metaclass=ABCMeta):
    """
    The base class for a scripted frame provider.

    A scripted frame provider allows you to provide custom stack frames for a
    thread, which can be used to augment or replace the standard unwinding
    mechanism. This is useful for:

    - Providing frames for custom calling conventions or languages
    - Reconstructing missing frames from crash dumps or core files
    - Adding diagnostic or synthetic frames for debugging
    - Visualizing state machines or async execution contexts

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.

    The constructor of this class sets up the following attributes:

    - ``input_frames`` (lldb.SBFrameList or None): The frame list to use as input
    - ``thread`` (lldb.SBThread or None): The thread this provider is attached to.
    - ``process`` (lldb.SBProcess or None): The process that owns the thread.
    - ``target`` (lldb.SBTarget or None): The target from the thread's process.
    - ``args`` (lldb.SBStructuredData or None): Dictionary-like structured data passed when the provider was registered.

    Example usage:

    .. code-block:: python

        from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider

        class MyFrameProvider(ScriptedFrameProvider):
            def __init__(self, input_frames, args):
                super().__init__(input_frames, args)

            @staticmethod
            def get_description():
                return "Show each frame twice"

            def get_frame_at_index(self, index):
                # Duplicate every frame
                return int(index / 2)

        def __lldb_init_module(debugger, internal_dict):
            debugger.HandleCommand(f"target frame-provider register -C {__name__}.MyFrameProvider")

        if __name__ == '__main__':
            print("This script should be loaded from LLDB using `command script import <filename>`")

    You can register your frame provider either via the CLI command ``target frame-provider register`` or
    via the API ``SBThread.RegisterScriptedFrameProvider``.
    """

    @staticmethod
    def applies_to_thread(thread):
        """Determine if this frame provider should be used for a given thread.

        This static method is called before creating an instance of the frame
        provider to determine if it should be applied to a specific thread.
        Override this method to provide custom filtering logic.

        Args:
            thread (lldb.SBThread): The thread to check.

        Returns:
            bool: True if this frame provider should be used for the thread,
                False otherwise. The default implementation returns True for
                all threads.

        Example:

        .. code-block:: python

            @staticmethod
            def applies_to_thread(thread):
                # Only apply to thread 1
                return thread.GetIndexID() == 1
        """
        return True

    @staticmethod
    @abstractmethod
    def get_description():
        """Get a description of this frame provider.

        This method should return a human-readable string describing what
        this frame provider does. The description is used for debugging
        and display purposes.

        Returns:
            str: A description of the frame provider.

        Example:

        .. code-block:: python

            @staticmethod
            def get_description(self):
                return "Crash log frame provider for thread 1"
        """
        pass

    @staticmethod
    def get_priority():
        """Get the priority of this frame provider.

        This static method is called to determine the evaluation order when
        multiple frame providers could apply to the same thread. Lower numbers
        indicate higher priority (like Unix nice values).

        Returns:
            int or None: Priority value where 0 is highest priority.
                Return None for default priority (UINT32_MAX - lowest priority).

        Example:

        .. code-block:: python

            @staticmethod
            def get_priority():
                # High priority - runs before most providers
                return 10

            @staticmethod
            def get_priority():
                # Default priority - runs last
                return None
        """
        return None  # Default/lowest priority

    def __init__(self, input_frames, args):
        """Construct a scripted frame provider.

        Args:
            input_frames (lldb.SBFrameList): The frame list to use as input.
                This allows you to access frames by index. The frames are
                materialized lazily as you access them.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted frame provider.
        """
        self.input_frames = None
        self.args = None
        self.thread = None
        self.target = None
        self.process = None

        if isinstance(input_frames, lldb.SBFrameList) and input_frames.IsValid():
            self.input_frames = input_frames
            self.thread = input_frames.GetThread()
            if self.thread and self.thread.IsValid():
                self.process = self.thread.GetProcess()
                if self.process and self.process.IsValid():
                    self.target = self.process.GetTarget()

        if isinstance(args, lldb.SBStructuredData) and args.IsValid():
            self.args = args

    @abstractmethod
    def get_frame_at_index(self, index):
        """Get a single stack frame at the given index.

        This method is called lazily when a specific frame is needed in the
        thread's backtrace (e.g., via the 'bt' command). Each frame is
        requested individually as needed.

        Args:
            index (int): The frame index to retrieve (0 for youngest/top frame).

        Returns:
            ScriptedFrame, integer, Dict or None: An object describing the stack
               stack frame, or None if no frame exists at this index.

            An integer represents the corresponding input frame index to reuse,
            in case you want to just forward an frame from the ``input_frames``.

            Returning a ScriptedFrame object injects artificial frames giving
            you full control over the frame behavior.

            Returning a dictionary also injects an artificial frame, but with
            less control over the frame behavior. The dictionary must contain:

            - idx (int): The synthetic frame index (0 for youngest/top frame)
            - pc (int): The program counter address for the synthetic frame

        Example:

        .. code-block:: python

            def get_frame_at_index(self, index):
                # Return None when there are no more frames
                if index >= self.total_frames:
                    return None

                # Re-use an input frame by returning its index
                if self.should_use_input_frame(index):
                    return index  # Returns input frame at this index

                # Or create a custom frame dictionary
                if index == 0:
                    return {
                        "idx": 0,
                        "pc": 0x100001234,
                    }

                return None

        Note:
            The frames are indexed from 0 (youngest/top) to N (oldest/bottom).
            This method will be called repeatedly with increasing indices until
            None is returned.
        """
        pass
