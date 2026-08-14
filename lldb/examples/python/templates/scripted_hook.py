from abc import ABCMeta, abstractmethod

import lldb


class ScriptedHook(metaclass=ABCMeta):
    """
    The base class for a scripted target hook.

    A single `ScriptedHook` subclass backs both `target hook add -P` and
    `target stop-hook add -P`. `handle_stop` is required so a hook can
    always be attached as a stop-hook; `handle_module_loaded` and
    `handle_module_unloaded` are optional and only called for hooks
    registered via `target hook add -P`.
    """

    @abstractmethod
    def __init__(self, target, args):
        """Construct a scripted hook.

        Args:
            target (lldb.SBTarget): The target owning this hook.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted hook.
        """
        self.target = target
        self.args = args

    def handle_module_loaded(self, stream):
        """Called whenever a module is loaded into the target.

        Args:
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.
        """
        pass

    def handle_module_unloaded(self, stream):
        """Called whenever a module is unloaded from the target.

        Args:
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.
        """
        pass

    @abstractmethod
    def handle_stop(self, exe_ctx, stream):
        """Called whenever the process stops, before control is returned to
        the user.

        Args:
            exe_ctx (lldb.SBExecutionContext): The execution context at the
                point of the stop.
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.

        Returns:
            bool: `True` if the process should stop and control should be
            returned to the user, `False` if the process should keep running.
        """
        pass
