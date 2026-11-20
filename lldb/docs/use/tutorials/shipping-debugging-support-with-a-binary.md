# Shipping Debugging Support with a Binary

A binary can bring its own debugging support: data formatters for its types,
commands to inspect it, or the breakpoints and stop hooks that every debug
session of it needs. You put that support in a Python script that LLDB loads
along with the binary, for each target that loads it. The script usually ships
in the binary's dSYM bundle:
[Auto-Loading Scripts from Code-Signed dSYM Bundles](auto-loading-scripts-from-dsyms.md)
explains where it goes and when LLDB loads it.

This tutorial explains the functions the script can define, when LLDB calls
each one, and how to write it so that it works when LLDB loads it more than
once. LLDB calls a binary a module, which is why these functions have module in
their names.

## The functions LLDB calls

| Function | Arguments | When LLDB calls it | Use it for |
|----------|-----------|--------------------|------------|
| `__lldb_init_module` | `debugger: lldb.SBDebugger` `internal_dict: dict` | Each time the script is loaded: once for each target that loads the module, and each time you import it with `command script import`. | Setup that belongs to the debugger: commands, data formatters, settings that apply to every target. |
| `__lldb_module_added_to_target` | `target: lldb.SBTarget` `internal_dict: dict` | Once for each target, when the module is added to it. The module may not have a load address yet, and there may be no process. | Setup that belongs to the target you were given: breakpoints, stop hooks, frame providers. |

Both are optional. A script that only adds commands and formatters needs just
`__lldb_init_module`. A script that uses both looks like this:

```python3
import lldb


def __lldb_init_module(debugger, internal_dict):
    # Commands and formatters work in every target.
    debugger.HandleCommand(
        "type summary add --summary-string '${var.name}' MyApp::Widget"
    )


def __lldb_module_added_to_target(target, internal_dict):
    # Breakpoints and stop hooks belong to the target that loaded the module.
    target.BreakpointCreateByName("my_app_assertion_failed")
```

## When LLDB calls them

Each time LLDB loads the script for a target, it imports the script as a Python
module, or reloads it if it was already imported, which runs its top-level code
again. Then it calls `__lldb_init_module`, and then
`__lldb_module_added_to_target` with that target. If `__lldb_init_module`
raises an exception, LLDB reports it and doesn't call
`__lldb_module_added_to_target`.

LLDB does this once for each target that loads the module, as soon as the
module is added to the target. That's usually before the module has a load
address, and often before there's a process: "Waiting for the process to stop"
below explains how to do work that needs the process.

`command script import` also runs `__lldb_init_module`, but there's no target to
pass, so LLDB doesn't call `__lldb_module_added_to_target`. A script that relies
on it has to be loaded with its binary.

If you edit the script, a new target picks up the change, but a target that
already loaded it doesn't. To reload it there, import it with
`command script import`.

## Keeping state across loads

When a second target loads the same module, LLDB loads the script again, which
reloads the Python module and runs its top-level code again. A variable that the
module sets at top level starts over each time. If it has to survive, for
example a set of the targets you already set up, create it only when it doesn't
exist yet:

```python3
if "configured_targets" not in globals():
    configured_targets = set()
```

## Modules that your script imports

LLDB only calls these functions on the script it loaded with the binary. If that script
imports other modules, with `import` or `command script import`, LLDB doesn't
call their `__lldb_module_added_to_target`. It doesn't know that they belong to
the module, or in which order they need to run.

If your script is a small loader that pulls in the rest of your tools, forward
the call yourself, in the order your modules need:

```python3
import os
import sys

TOOLS = ["memory_tools", "thread_tools"]


def __lldb_init_module(debugger, internal_dict):
    directory = os.path.dirname(__file__)
    for name in TOOLS:
        path = os.path.join(directory, name + ".py")
        debugger.HandleCommand(f"command script import {path}")


def __lldb_module_added_to_target(target, internal_dict):
    for name in TOOLS:
        hook = getattr(sys.modules[name], "__lldb_module_added_to_target", None)
        if hook:
            hook(target, internal_dict)
```

## Waiting for the process to stop

LLDB calls `__lldb_module_added_to_target` when the module joins the target, so
the script usually can't read the process from there: the module has no load
address yet, there's no process, or, for a core file or an attach, the process
is still being loaded. To do something once the process has stopped, add a
one-shot stop hook with `--at-initial-stop true` from
`__lldb_module_added_to_target`. LLDB runs it at the first stop it can, then
deletes it.

For example, this script reads its module's Mach-O header from memory, because
the information it needs is only there and not in the binary on disk:

```python3
import struct

import lldb

MODULE_NAME = "MyApp"


class ReadMachOHeader:
    def __init__(self, target, extra_args, internal_dict):
        self.done = False

    def handle_stop(self, exe_ctx, stream):
        # LLDB runs the hook for every thread that stopped, and we only need
        # to read the header once.
        if self.done:
            return True
        self.done = True

        target = exe_ctx.GetTarget()
        module = target.module[MODULE_NAME]
        header = module.GetObjectFileHeaderAddress().GetLoadAddress(target)
        if header == lldb.LLDB_INVALID_ADDRESS:
            # Too early: the module isn't loaded yet. Try again at the next
            # stop.
            add_header_hook(target)
            return True

        error = lldb.SBError()
        data = exe_ctx.GetProcess().ReadMemory(header, 32, error)
        if error.Fail():
            stream.Print(f"{MODULE_NAME}: could not read the Mach-O header: {error}\n")
            return True

        _, _, _, _, ncmds, sizeofcmds, _, _ = struct.unpack("<8I", data)
        stream.Print(f"{MODULE_NAME}: {ncmds} load commands, {sizeofcmds} bytes\n")
        return True


def add_header_hook(target):
    result = lldb.SBCommandReturnObject()
    target.GetDebugger().GetCommandInterpreter().HandleCommand(
        f"target stop-hook add --one-shot true --at-initial-stop true "
        f"-P {__name__}.ReadMachOHeader",
        lldb.SBExecutionContext(target),
        result,
    )


def __lldb_module_added_to_target(target, internal_dict):
    add_header_hook(target)
```

A few things in this example matter:

- `HandleCommand` gets an `SBExecutionContext` for the target that
  `__lldb_module_added_to_target` was given, so the stop hook goes on that
  target, not on whichever target is selected.
- `--at-initial-stop true` lets the stop hook run at the stop LLDB is handling
  when it loads the script: when an attach completes, when a core file or a
  scripted process has finished loading, or, if the process already stopped,
  as soon as `__lldb_module_added_to_target` returns. Without it, the hook
  waits for the process to stop again, which a core file never does.
- Only the stop hooks that `__lldb_module_added_to_target` adds run as soon as
  it returns. If the process already stopped, a stop hook that
  `__lldb_init_module` or any other script adds first runs at the next stop,
  even with `--at-initial-stop true`.
- When LLDB loads the script with the target, before there's a process, the
  hook runs at the process's first stop. For a launch, that stop comes before
  the loader has run, so the module doesn't have a load address yet, and the
  hook adds itself again for the next stop.
- `handle_stop` returns `True`: the hook only reads, and the process should
  stay stopped. Returning `False` would ask LLDB to continue.
- At the stop where it runs, LLDB calls `handle_stop` for every thread that
  stopped, so the hook remembers that it already read the header.
- Because the hook is one-shot, LLDB deletes it after that stop, so it doesn't
  run at later stops and doesn't vote on whether they continue. It runs for one
  process: if you relaunch and need it again, add it again.

Read whatever you need from the stop hook, but don't resume, step, detach or
kill the process from it: let the user or a command decide what the process
does next.
