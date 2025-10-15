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
    def get_stackframes(self, real_frames):
        """Get the list of stack frames to provide.

        This method is called when the thread's backtrace is requested
        (e.g., via the 'bt' command). The returned frames will be integrated
        with the real frames according to the mode returned by get_mode().

        Args:
            real_frames (lldb.SBFrameList): The actual unwound frames from the
                thread's normal unwinder. This allows you to iterate, filter,
                and selectively replace frames. The frames are materialized
                lazily as you access them.

        Returns:
            List[Dict]: A list of frame dictionaries, where each dictionary
                describes a single stack frame. Each dictionary should contain:

            Required fields:
            - idx (int): The frame index (0 for innermost/top frame)
            - pc (int): The program counter address for this frame

            Alternatively, you can return a list of ScriptedFrame objects
            for more control over frame behavior.

        Example:

        .. code-block:: python

            def get_stackframes(self, real_frames):
                frames = []

                # Iterate over real frames and filter/augment them
                for i, frame in enumerate(real_frames):
                    if self.should_include_frame(frame):
                        frames.append({
                            "idx": i,
                            "pc": frame.GetPC(),
                        })

                # Or create custom frames
                frames.append({
                    "idx": 0,
                    "pc": 0x100001234,
                })

                return frames

        Note:
            The frames are indexed from 0 (innermost/newest) to N (outermost/oldest).
        """
        pass
