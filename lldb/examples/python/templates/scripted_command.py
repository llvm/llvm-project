from abc import ABCMeta, abstractmethod
from typing import Optional

import lldb


class ScriptedCommand(metaclass=ABCMeta):
    """
    The base class for a scripted (raw) command.

    A raw command receives the unparsed argument string exactly as the user
    typed it, and is responsible for any parsing it needs. For a command
    with a table-driven option/argument parser, see `ParsedCommand` instead.
    Register it with `command script add -c <ClassName> ...`.

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.
    """

    def __init__(self, debugger: lldb.SBDebugger):
        """Construct a scripted command.

        Args:
            debugger (lldb.SBDebugger): The debugger this command is being
                added to.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        debugger: lldb.SBDebugger,
        args: str,
        exe_ctx: lldb.SBExecutionContext,
        result: lldb.SBCommandReturnObject,
    ) -> None:
        """Execute the command.

        Args:
            debugger (lldb.SBDebugger): The debugger the command runs
                against.
            args (str): The raw, unparsed argument string.
            exe_ctx (lldb.SBExecutionContext): The execution context.
            result (lldb.SBCommandReturnObject): Write command output/errors
                here.
        """
        pass

    def get_short_help(self) -> Optional[str]:
        """A one-line description shown by `help`.

        Returns:
            str: The short help string.
        """
        pass

    def get_long_help(self) -> Optional[str]:
        """The full help text shown by `help <command>`.

        Returns:
            str: The long help string.
        """
        pass

    def get_flags(self) -> int:
        """Command flags (a bitmask of `lldb.eCommandRequires*`/
        `lldb.eCommandProcessMustBe*` etc.) controlling when this command is
        available.

        Returns:
            int: The flags bitmask. Defaults to 0 (no restrictions).
        """
        return 0

    def get_repeat_command(self, command: str) -> Optional[str]:
        """Customize what runs when the user presses Enter to repeat this
        command.

        Args:
            command (str): The command line that was run.

        Returns:
            str: The command line to run on repeat. Defaults to `None`,
            meaning repeat the original command unmodified.
        """
        pass
