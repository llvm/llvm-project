from abc import ABCMeta, abstractmethod

import lldb


class ScriptedBreakpointResolver(metaclass=ABCMeta):
    """
    The base class for a scripted breakpoint resolver.
    """

    @abstractmethod
    def __init__(self, bkpt, args):
        """Construct a scripted breakpoint resolver.

        Args:
            bkpt (lldb.SBBreakpoint): The breakpoint owning this resolver.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted breakpoint.
        """
        self.bkpt = bkpt
        self.args = args

    def overrides_resolver(self, resolver_data):
        """Decide, from the incoming resolver options, whether this
        breakpoint should have its resolver replaced by this class. When
        `True`, this class's `__callback__` picks locations for this
        breakpoint instead of the original resolver's. Search depth is
        unaffected either way and still comes from `__get_depth__`.

        Args:
            resolver_data (lldb.SBStructuredData): The resolver options
                passed in when the breakpoint was created.

        Returns:
            bool: `True` if this class's `__callback__` should be used in
            place of the original resolver's, `False` to leave the
            original resolver in charge.
        """
        return False

    def set_breakpoint(self, bkpt):
        """Called once the underlying breakpoint has been fully created and
        associated to this resolver.

        Args:
            bkpt (lldb.SBBreakpoint): The breakpoint owning this resolver.
        """
        pass

    @abstractmethod
    def __callback__(self, sym_ctx):
        """Called once per symbol context matched by the search depth
        returned by `__get_depth__`. Set breakpoint locations here by calling
        `AddLocation` on the resolver's breakpoint.

        Args:
            sym_ctx (lldb.SBSymbolContext): The symbol context to inspect.
        """
        pass

    def __get_depth__(self):
        """The search depth at which `__callback__` will be called.

        Returns:
            lldb.SBSearchDepth: One of the `lldb.eSearchDepth*` values.
            Defaults to `lldb.eSearchDepthModule`.
        """
        return lldb.eSearchDepthModule

    def get_short_help(self):
        """A one-line description of this resolver, shown by `breakpoint list`.

        Returns:
            str: The short help string.
        """
        pass

    def was_hit(self, frame, bp_loc):
        """Called when a location owned by this resolver is hit, to allow
        overriding which location is reported as hit.

        Args:
            frame (lldb.SBFrame): The frame where the breakpoint was hit.
            bp_loc (lldb.SBBreakpointLocation): The breakpoint location that
                was hit.

        Returns:
            lldb.SBBreakpointLocation: The location to report as hit.
            Defaults to `bp_loc`.
        """
        return bp_loc

    def get_location_description(self, bp_loc, level):
        """Customize the description used when printing a breakpoint location
        owned by this resolver.

        Args:
            bp_loc (lldb.SBBreakpointLocation): The breakpoint location to
                describe.
            level (lldb.DescriptionLevel): The level of detail requested.

        Returns:
            str: The description for the location.
        """
        pass
