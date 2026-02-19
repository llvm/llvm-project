from abc import ABCMeta, abstractmethod
import os

import lldb


class ScriptedSymbolLocator(metaclass=ABCMeta):
    """
    The base class for a scripted symbol locator.

    Most of the base class methods are optional and return ``None`` to fall
    through to LLDB's default resolution. Override only the methods you need.

    Configuration::

        (lldb) command script import /path/to/my_locator.py
        (lldb) target symbols scripted register -C my_locator.MyLocator \\
                   [-k key -v value ...]
    """

    @abstractmethod
    def __init__(self, exe_ctx, args):
        """Construct a scripted symbol locator.

        Args:
            exe_ctx (lldb.SBExecutionContext): The execution context for
                the scripted symbol locator.
            args (lldb.SBStructuredData): A Dictionary holding arbitrary
                key/value pairs used by the scripted symbol locator.
        """
        target = None
        self.target = None
        self.args = None
        if isinstance(exe_ctx, lldb.SBExecutionContext):
            target = exe_ctx.target
        if isinstance(target, lldb.SBTarget) and target.IsValid():
            self.target = target
            self.dbg = target.GetDebugger()
        if isinstance(args, lldb.SBStructuredData) and args.IsValid():
            self.args = args

    def locate_source_file(self, module, original_source_file):
        """Locate the source file for a given module.

        Called when LLDB resolves source file paths during stack frame
        display, breakpoint resolution, or source listing. This is the
        primary method for implementing source file remapping based on
        build IDs.

        The module is a fully loaded ``SBModule`` (not an ``SBModuleSpec``),
        so you can access its UUID, file path, platform file path,
        symbol file path, sections, and symbols.

        Results are cached per (module UUID, source file) pair, so this
        method is called at most once per unique combination.

        Args:
            module (lldb.SBModule): The loaded module containing debug
                info. Use ``module.GetUUIDString()`` to get the build ID
                for looking up the correct source revision.
            original_source_file (str): The original source file path
                as recorded in the debug info.

        Returns:
            lldb.SBFileSpec: The resolved file spec, or None to fall
                through to LLDB's default source resolution.
        """
        return None


class LocalCacheSymbolLocator(ScriptedSymbolLocator):
    """Example locator that resolves source files from a local cache directory.

    Demonstrates how to subclass ``ScriptedSymbolLocator`` to implement
    custom source file resolution. This locator looks up source files
    in a local directory structure organized by build ID (UUID)::

        <cache_dir>/
            <uuid>/
                src/
                    main.cpp
                    ...

    Usage::

        (lldb) command script import scripted_symbol_locator
        (lldb) target symbols scripted register \\
                   -C scripted_symbol_locator.LocalCacheSymbolLocator \\
                   -k cache_dir -v "/path/to/cache"
        (lldb) target create --core /path/to/minidump.dmp
        (lldb) bt

    The locator searches for:
      - Source files:   ``<cache_dir>/<uuid>/src/<basename>``
    """

    cache_dir = None

    def __init__(self, exe_ctx, args):
        super().__init__(exe_ctx, args)

        # Allow cache_dir to be set via structured data args.
        if self.args:
            cache_dir_val = self.args.GetValueForKey("cache_dir")
            if cache_dir_val and cache_dir_val.IsValid():
                val = cache_dir_val.GetStringValue(256)
                if val:
                    LocalCacheSymbolLocator.cache_dir = val

    def _get_cache_path(self, uuid_str, *components):
        """Build a path under the cache directory for a given UUID.

        Args:
            uuid_str (str): The module's UUID string.
            *components: Additional path components (e.g., filename).

        Returns:
            str: The full path, or None if cache_dir is not set or the
                UUID is empty.
        """
        if not self.cache_dir or not uuid_str:
            return None
        return os.path.join(self.cache_dir, uuid_str, *components)

    def locate_source_file(self, module, original_source_file):
        """Look up source files under ``<cache_dir>/<uuid>/src/``."""
        uuid_str = module.GetUUIDString()
        basename = os.path.basename(original_source_file)
        path = self._get_cache_path(uuid_str, "src", basename)
        if path and os.path.exists(path):
            return lldb.SBFileSpec(path, True)
        return None
