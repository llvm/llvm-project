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

    Example usage:

    .. code-block:: python

        # Attach a frame provider to a thread
        thread = process.GetSelectedThread()
        error = thread.SetScriptedFrameProvider(
                        "my_module.MyFrameProvider",
                        lldb.SBStructuredData()
        )
    """

    @abstractmethod
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
            Dict or None: A frame dictionary describing the stack frame, or None
                if no frame exists at this index. The dictionary should contain:

            Required fields:
            - idx (int): The synthetic frame index (0 for youngest/top frame)
            - pc (int): The program counter address for the synthetic frame

            Alternatively, you can return:
            - A ScriptedFrame object for full control over frame behavior
            - An integer representing an input frame index to reuse
            - None to indicate no more frames exist

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
