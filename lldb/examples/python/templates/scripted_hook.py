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

    target: lldb.SBTarget
    args: lldb.SBStructuredData

    @abstractmethod
    def __init__(self, target: lldb.SBTarget, args: lldb.SBStructuredData):
        """Construct a scripted hook.

        Args:
            target (lldb.SBTarget): The target owning this hook.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted hook.
        """
        self.target = target
        self.args = args

    def handle_module_loaded(self, stream: lldb.SBStream) -> None:
        """Called whenever a module is loaded into the target.

        Args:
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.
        """
        pass

    def handle_module_unloaded(self, stream: lldb.SBStream) -> None:
        """Called whenever a module is unloaded from the target.

        Args:
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.
        """
        pass

    @abstractmethod
    def handle_stop(
        self, exe_ctx: lldb.SBExecutionContext, stream: lldb.SBStream
    ) -> bool:
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


    def handle_resolve_addr(
        self, load_addr: int, stream: lldb.SBStream) -> lldb.SBAddress:
        """Called whenever the target is not able to resolve a load address to
        a section offset address.

        Clients can implement a JIT loader plugin using this method. Anytime the
        target fails to resolve an address, this method will be called. The
        function can find the file for the module that contains the address,
        load it into the target, and return the resolved address. If the
        address cannot be resolved, return a default construct lldb.SBAddress.

        Args:
            load_addr (int): The load address to attempt to resolve.
            stream (lldb.SBStream): The stream to which the hook can write
                output that will be reported to the user.

        Returns:
            lldb.SBAddress: If the address was resolved, return a SBAddress
            that has been resolved to a section offset address, or return a
            default constructed SBAddress if the address could not be resolved.
        """
        pass
