# Auto-Loading Scripts from Code-Signed dSYM Bundles

A dSYM bundle can carry a Python script that LLDB loads with the module it
describes: data formatters for the module's types, commands to inspect it, or
setup for the targets that load it.
[Shipping Debugging Support with a Binary](shipping-debugging-support-with-a-binary.md)
explains how to write the script. This tutorial explains where to put the
script, when LLDB loads it on its own, and how to sign the bundle so that it
does.

## Where the script goes

LLDB looks for the script in the dSYM bundle, in `Contents/Resources/Python`,
named after the module:

```
MyApp.dSYM/
  Contents/
    Info.plist
    Resources/
      DWARF/
        MyApp
      Python/
        MyApp.py
```

The script's name is the module's file name, changed to a valid Python module
name: LLDB replaces `.`, spaces and `-` with `_`, replaces `+` with `x`, and adds
a leading `_` to a name that is a Python keyword. If there is no script with
that name, LLDB strips the last extension from the file name and tries again.
For example, for a module named `a.out`, LLDB looks for `a_out.py`, then for
`a.py`.

A framework works the same way, through its dSYM:
`MyKit.framework.dSYM/Contents/Resources/Python/MyKit.py` is loaded with
`MyKit.framework/MyKit`.

## Packaging the script with dsymutil

Instead of copying the script into the dSYM after the build, let `dsymutil`
do it when it creates the bundle. `--embed-resource <source>=<destination>`
copies a file or a directory into the bundle, at a destination relative to
`Contents/Resources`:

```
$ dsymutil MyApp -o MyApp.dSYM --embed-resource scripts/MyApp.py=Python/MyApp.py
```

A directory is copied with its contents, so a script that imports helper
modules can ship them next to it:

```
$ dsymutil MyApp -o MyApp.dSYM --embed-resource scripts=Python
```

The option can be repeated, and doesn't work with `--flat`, which doesn't
create a bundle.

## The safe auto-load locations

When the dSYM has no script for a module, or there is no dSYM, LLDB looks in
its safe auto-load locations, a list of directories that ship with LLDB or
with the SDK rather than with the module:

- The SDK that matches the target, `$SDKROOT/usr/share/lldb`, on Darwin.
- The directories LLDB was configured with, through the
  `LLDB_SAFE_AUTO_LOAD_PATHS` CMake option.

In each location, the scripts for a module go in a directory named after it,
without its extension: `$SDKROOT/usr/share/lldb/MyApp/MyApp.py`. LLDB uses the
first location that has a directory for the module, even if the script it
expects isn't there, and the settings below decide whether it loads the
script, as they do for a dSYM.

## When LLDB loads the script on its own

`target.load-script-from-symbol-file` decides what LLDB does with the scripts
it finds:

| Value | What LLDB does |
|-------|----------------|
| `trusted` (the default) | Loads the script if the dSYM bundle has a trusted code signature, and warns about the script otherwise. |
| `true` | Loads every script it finds. |
| `warn` | Never loads scripts, and warns about each one it finds. |
| `false` | Never loads scripts, and says nothing. |

`target.auto-load-scripts-for-modules` overrides that setting for the modules
you name, with the same values:

```
(lldb) settings set target.auto-load-scripts-for-modules MyApp=true
```

Use the module's file name without its extension. A per-module entry is the
narrowest way to allow a script that isn't signed: it allows one module's
scripts instead of every script in every dSYM.

Set both before the target loads the module, for example in `~/.lldbinit`. LLDB
decides when the module loads, and changing the settings later doesn't load the
scripts of modules that are already in the target.

## What makes a dSYM bundle trusted

A bundle is trusted when its code signature is valid and chains to a
certificate authority that the system trusts. In code signing terms, it
satisfies the `anchor trusted` requirement. You can check a bundle the same way
LLDB does:

```
$ codesign --verify --verbose -R="anchor trusted" MyApp.dSYM
MyApp.dSYM: valid on disk
MyApp.dSYM: satisfies its Designated Requirement
MyApp.dSYM: explicit requirement satisfied
```

An ad hoc signature, `codesign --sign -`, never satisfies it, because it doesn't
chain to any authority.

## Signing a dSYM bundle

Sign the bundle with an identity whose certificate chains to a trusted
authority, such as an Apple Development or Developer ID certificate:

```
$ codesign --force --sign "Apple Development: Jane Appleseed (ABCDE12345)" MyApp.dSYM
```

The signature seals the whole bundle, the script included. Editing the script
or the DWARF afterwards breaks the signature, and LLDB goes back to warning
about the script. Sign the bundle as the last step of the build, after
`dsymutil` and after copying the script in, and sign it again after every
change.

A signature only helps where its certificate is trusted. If you sign with a
certificate from a private authority, such as a company's internal signing
service, a machine that doesn't trust that authority, for example one that
downloaded the dSYM from a bug report, treats the bundle as untrusted. On those
machines, allow the module with `target.auto-load-scripts-for-modules`.

## When LLDB doesn't load the script

When the settings don't let LLDB load a script on its own, LLDB asks you
instead. In an interactive session, it lists those scripts in a picker once the
process stops. Trusted scripts are checked, and LLDB loads the ones you leave
checked, for this debug session:

[![asciicast](https://asciinema.org/a/vEHGCf50bSbSwvPZ.svg)](https://asciinema.org/a/vEHGCf50bSbSwvPZ)

When the picker isn't available, for example when you run `lldb --batch`, LLDB
falls back to printing a warning for each script, with the command that loads
it:

```
warning: 'MyApp' contains an untrusted debug script. To run this script in this debug session:

    command script import "/path/to/MyApp.dSYM/Contents/Resources/Python/MyApp.py"

To run all discovered debug scripts in this session:

    settings set target.load-script-from-symbol-file true
```

Importing the script by hand doesn't call `__lldb_module_added_to_target`, as
[Shipping Debugging Support with a Binary](shipping-debugging-support-with-a-binary.md)
explains. If your script relies on it, allow the module with
`target.auto-load-scripts-for-modules` and create the target again.

To see what LLDB found and why it did or didn't load a script, enable the
module log before creating the target:

```
(lldb) log enable lldb module
```
