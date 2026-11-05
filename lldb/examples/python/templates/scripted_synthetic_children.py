from abc import ABCMeta, abstractmethod
from typing import Optional

import lldb


class ScriptedSyntheticChildren(metaclass=ABCMeta):
    """
    The base class for a scripted synthetic children provider.

    A synthetic children provider allows you to customize how a value is
    expanded into children when displayed (e.g. `frame variable`, `bt`).
    Register it with `type synthetic add -l <ClassName> ...`.

    Most of the base class methods are `@abstractmethod` that need to be
    overwritten by the inheriting class.
    """

    valobj: lldb.SBValue

    def __init__(self, valobj: lldb.SBValue):
        """Construct a scripted synthetic children provider.

        Args:
            valobj (lldb.SBValue): The value this provider generates children
                for.
        """
        self.valobj = valobj

    @abstractmethod
    def num_children(self) -> int:
        """The number of children this value has.

        This can optionally take a second `max_count` parameter (i.e.
        `def num_children(self, max_count)`) if computing the exact count is
        expensive; in that case return `max_count` once at least that many
        children are known to exist.

        Returns:
            int: The number of children.
        """
        pass

    @abstractmethod
    def get_child_at_index(self, index: int) -> Optional[lldb.SBValue]:
        """Get the child at the given index.

        Args:
            index (int): The index of the child to return.

        Returns:
            lldb.SBValue: The value for the child at this index, or `None` if
            there is no child at this index.
        """
        pass

    def get_child_index(self, name: str) -> Optional[int]:
        """Get the index of the child with the given name.

        Args:
            name (str): The name of the child to look up.

        Returns:
            int: The index of the child with this name, or `None`/a negative
            value if no such child exists. Defaults to a linear search over
            `get_child_at_index`/`num_children`.
        """
        pass

    def update(self) -> bool:
        """Called when the value backing this provider may have changed
        (e.g. after a `continue`), giving the provider a chance to refresh
        any cached state.

        Returns:
            bool: `True` if the previously computed children can be reused,
            `False` if they should be recomputed. Defaults to `False`.
        """
        return False

    def has_children(self) -> bool:
        """Whether this value might have children, without necessarily
        computing them. Used as a cheap check to decide whether to show an
        expansion arrow in graphical frontends, for example.

        Returns:
            bool: `True` if this value might have children, `False`
            otherwise. Defaults to `True`.
        """
        return True

    def get_value(self) -> Optional[lldb.SBValue]:
        """Make this a value-providing synthetic children provider: the
        value returned here becomes the value for this `SBValue`, in place
        of the value backing it. None of the other methods on this class
        (`num_children`, `get_child_at_index`, `get_child_index`) are
        consulted, and the children of the original value are not shown.

        Returns:
            lldb.SBValue: The value to use instead of this value's own,
            or `None` to leave this value unaffected. Defaults to `None`.
        """
        return None

    def get_type_name(self) -> Optional[str]:
        """Override the type name shown for this synthetic value.

        Returns:
            str: The type name to display, or `None`/empty to keep the
            default. Defaults to `None`.
        """
        pass
