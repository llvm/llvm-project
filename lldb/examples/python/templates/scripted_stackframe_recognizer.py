from abc import ABCMeta
from typing import Optional

import lldb


class ScriptedStackFrameRecognizer(metaclass=ABCMeta):
    """
    The base class for a scripted stack frame recognizer.

    A frame recognizer inspects a stack frame at stop time and can:

    - attach recognized arguments to it (surfaced the same way as args
      for unrecognized frames);
    - hide the frame from backtraces;
    - override the stop description reported for it;
    - expose an exception object associated with it (e.g. a runtime's
      thrown value);
    - redirect auto-selection to a more relevant frame (e.g. bounce past
      a language-runtime trampoline).

    Register the recognizer with `frame recognizer add -l <ClassName> ...`.

    Every method is optional: a recognizer only implements the ones it
    cares about. A recognizer that just hides frames can implement only
    `should_hide`; one that surfaces exception info can implement only
    `get_exception` and `get_stop_description`.
    """

    def __init__(self):
        """Construct a scripted stack frame recognizer.

        Recognizers are constructed with no arguments and are shared across
        every frame they're asked to recognize.
        """
        pass

    def get_recognized_arguments(self, frame: lldb.SBFrame) -> list:
        """Get the arguments recognized for this frame.

        Args:
            frame (lldb.SBFrame): The frame to inspect.

        Returns:
            list of lldb.SBValue: The recognized arguments, or an empty list
            if none could be recognized.
        """
        return []

    def should_hide(self, frame: lldb.SBFrame) -> bool:
        """Whether this frame should be hidden when displaying backtraces.

        Args:
            frame (lldb.SBFrame): The frame to inspect.

        Returns:
            bool: `True` if this frame should be hidden, `False` otherwise.
            Defaults to `False`.
        """
        return False

    def select_most_relevant_frame(self, frame: lldb.SBFrame) -> Optional[lldb.SBFrame]:
        """Pick a different frame to auto-select when the process stops in
        this recognized frame. Useful when the recognized frame is trampoline
        code that the user probably didn't want to land in (e.g. a language
        runtime's exception-throw path).

        Args:
            frame (lldb.SBFrame): The recognized frame.

        Returns:
            lldb.SBFrame: The frame to select instead, or `None` to keep
            the default selection.
        """
        return None

    def get_exception(self, frame: lldb.SBFrame) -> Optional[lldb.SBValue]:
        """Get the exception value associated with this recognized frame,
        if any (e.g. the exception object at a language runtime's throw
        site).

        Args:
            frame (lldb.SBFrame): The recognized frame.

        Returns:
            lldb.SBValue: The exception value, or `None` if this frame
            doesn't correspond to an exception.
        """
        return None

    def get_stop_description(self, frame: lldb.SBFrame) -> str:
        """Get the stop description to surface for this recognized frame,
        replacing whatever description lldb would have printed.

        Args:
            frame (lldb.SBFrame): The recognized frame.

        Returns:
            str: The stop description, or the empty string to keep the
            default.
        """
        return ""
