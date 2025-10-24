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
    def __init__(self, thread, args):
        """Construct a scripted frame provider.

        Args:
            thread (lldb.SBThread): The thread for which to provide frames.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted frame provider.
        """
        self.thread = None
        self.args = None
        self.target = None
        self.process = None

        if isinstance(thread, lldb.SBThread) and thread.IsValid():
            self.thread = thread
            self.process = thread.GetProcess()
            if self.process and self.process.IsValid():
                self.target = self.process.GetTarget()

        if isinstance(args, lldb.SBStructuredData) and args.IsValid():
            self.args = args

    @abstractmethod
    def get_frame_at_index(self, real_frames, index):
        """Get a single stack frame at the given index.

        This method is called lazily when a specific frame is needed in the
        thread's backtrace (e.g., via the 'bt' command). Each frame is
        requested individually as needed.

        Args:
            real_frames (lldb.SBFrameList): The actual unwound frames from the
                thread's normal unwinder. This allows you to access real frames
                by index. The frames are materialized lazily as you access them.
            index (int): The frame index to retrieve (0 for innermost/top frame).

        Returns:
            Dict or None: A frame dictionary describing the stack frame, or None
                if no frame exists at this index. The dictionary should contain:

            Required fields:
            - idx (int): The frame index (0 for innermost/top frame)
            - pc (int): The program counter address for this frame

            Alternatively, you can return:
            - A ScriptedFrame object for full control over frame behavior
            - An integer representing a real frame index to reuse
            - None to indicate no more frames exist

        Example:

        .. code-block:: python

            def get_frame_at_index(self, real_frames, index):
                # Return None when there are no more frames
                if index >= self.total_frames:
                    return None

                # Re-use a real frame by returning its index
                if self.should_use_real_frame(index):
                    return index  # Returns real frame at this index

                # Or create a custom frame
                if index == 0:
                    return {
                        "idx": 0,
                        "pc": 0x100001234,
                    }

                # Filter out some real frames
                real_frame = real_frames[index]
                if self.should_include_frame(real_frame):
                    return {
                        "idx": index,
                        "pc": real_frame.GetPC(),
                    }

                return None

        Note:
            The frames are indexed from 0 (innermost/newest) to N (outermost/oldest).
            This method will be called repeatedly with increasing indices until
            None is returned.
        """
        pass
