from abc import ABCMeta, abstractmethod

import lldb


class ScriptedStringSummary(metaclass=ABCMeta):
    """
    The base class for a scripted string summary provider.

    A summary provider produces the one-line string shown next to a value in
    `frame variable`/`expression` output. Register it with
    `type summary add -l <ClassName> <TypeName>`.

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.
    """

    def __init__(self):
        """Construct a scripted summary provider.

        Summary providers are constructed with no arguments and are shared
        across every value they're asked to summarize.
        """
        pass

    @abstractmethod
    def get_summary(
        self, valobj: lldb.SBValue, options: lldb.SBTypeSummaryOptions
    ) -> str:
        """Get the summary string for a value.

        Args:
            valobj (lldb.SBValue): The value to summarize.
            options (lldb.SBTypeSummaryOptions): The options to use when
                producing the summary.

        Returns:
            str: The summary string for `valobj`.
        """
        pass
