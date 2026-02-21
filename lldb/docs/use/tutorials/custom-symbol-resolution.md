# Finding Symbols With a Scripted Symbol Locator

The **Scripted Symbol Locator** lets you write a Python class that tells LLDB
where to find source files for your debug targets. This is useful when your
build artifacts live in a custom location, such as a symbol server or a local
build-ID-indexed cache.

## Quick Start

1. **Write a locator class.** Create a Python file (e.g., `my_locator.py`)
   with a class that implements the methods you need:

   ```python
   import os
   import lldb
   from lldb.plugins.scripted_symbol_locator import ScriptedSymbolLocator

   class MyLocator(ScriptedSymbolLocator):
       def __init__(self, exe_ctx, args):
           super().__init__(exe_ctx, args)
           self.cache_dir = None
           if self.args and self.args.IsValid():
               d = self.args.GetValueForKey("cache_dir")
               if d and d.IsValid():
                   self.cache_dir = d.GetStringValue(4096)

       def locate_source_file(self, module, original_source_file):
           """Return the resolved file spec, or None to fall through."""
           if not self.cache_dir:
               return None
           uuid = module.GetUUIDString()
           basename = os.path.basename(original_source_file)
           candidate = os.path.join(self.cache_dir, uuid, "src", basename)
           if os.path.exists(candidate):
               return lldb.SBFileSpec(candidate, True)
           return None
   ```

2. **Import the script and register the locator on a target:**

   ```
   (lldb) command script import /path/to/my_locator.py
   (lldb) target symbols scripted register \
              -C my_locator.MyLocator \
              -k cache_dir -v /path/to/cache
   ```

3. **Debug normally.** When LLDB resolves source files for that target,
   your `locate_source_file` method will be called automatically.

## Available Methods

Your locator class must implement `__init__` and `locate_source_file`.

| Method | Called When |
|--------|------------|
| `locate_source_file(module, path)` | LLDB resolves a source file path in debug info |

### Method Signatures

```python
def __init__(self, exe_ctx: lldb.SBExecutionContext,
             args: lldb.SBStructuredData) -> None:
    ...

def locate_source_file(self, module: lldb.SBModule,
                       original_source_file: str) -> Optional[lldb.SBFileSpec]:
    ...
```

## Per-Target Registration

The scripted symbol locator is registered **per target**. Different targets
can use different locator classes or different arguments.

```
(lldb) target select 0
(lldb) target symbols scripted register -C my_locator.MyLocator \
           -k cache_dir -v /cache/project-a

(lldb) target select 1
(lldb) target symbols scripted register -C my_locator.MyLocator \
           -k cache_dir -v /cache/project-b
```

### Commands

| Command | Description |
|---------|-------------|
| `target symbols scripted register -C <class> [-k <key> -v <value> ...]` | Register a locator |
| `target symbols scripted clear` | Remove the locator from the current target |
| `target symbols scripted info` | Show the current locator class |

### SB API

You can also register locators programmatically:

```python
import lldb

error = target.RegisterScriptedSymbolLocator(
    "my_locator.MyLocator", args)
# args is an SBStructuredData dictionary

target.ClearScriptedSymbolLocator()
```

## Caching

Source file resolutions are cached per `(module UUID, source file path)` pair
within each target. The cache is cleared when:

- A new locator is registered (via `register`)
- The locator is cleared (via `clear`)

This means your `locate_source_file` method is called at most once per
unique `(UUID, path)` combination.

## Base Class Template

LLDB ships a base class template at `lldb.plugins.scripted_symbol_locator`.
You can import and subclass it:

```python
from lldb.plugins.scripted_symbol_locator import ScriptedSymbolLocator

class MyLocator(ScriptedSymbolLocator):
    def __init__(self, exe_ctx, args):
        super().__init__(exe_ctx, args)

    def locate_source_file(self, module, original_source_file):
        # Your implementation here
        return None
```

The base class handles extracting the target and args from the execution
context. See `lldb/examples/python/templates/scripted_symbol_locator.py`
for the full template with docstrings.

## Listing Scripting Extensions

To see all registered scripting extensions (including symbol locators):

```
(lldb) scripting extension list
```
