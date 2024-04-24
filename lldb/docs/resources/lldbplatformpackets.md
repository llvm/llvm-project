# LLDB Platform Packets

This is a list of the packets that an lldb platform server
needs to implement for the lldb testsuite to be run on a remote
target device/system.

These are almost all lldb extensions to the gdb-remote serial
protocol. Many of the `vFile:` packets are also described in the "Host
I/O Packets" detailed in the gdb-remote protocol documentation,
although the lldb platform extensions include packets that are not
defined there (`vFile:size:`, `vFile:mode:`, `vFile:symlink`, `vFile:chmod:`).

Most importantly, the flags that LLDB passes to `vFile:open:` are
incompatible with the flags that GDB specifies.

* [](./lldbgdbremote.md#qstartnoackmode)
* [](./lldbgdbremote.md#qhostinfo)
* [](./lldbgdbremote.md#qmoduleinfo-module-path-arch-triple)
* [](./lldbgdbremote.md#qgetworkingdir)
* [](./lldbgdbremote.md#qsetworkingdir-ascii-hex-path)
* [](./lldbgdbremote.md#qplatform-mkdir)
* [](./lldbgdbremote.md#qplatform-shell)
* [](./lldbgdbremote.md#qlaunchgdbserver-platform-extension)
* [](./lldbgdbremote.md#qkillspawnedprocess-platform-extension)
* [](./lldbgdbremote.md#qprocessinfopid-pid-platform-extension)
  * It is likely that you only need to support the `pid` and `name` fields.
* [](./lldbgdbremote.md#qprocessinfopid-pid-platform-extension)
  * The lldb test suite currently only uses `name_match:equals` and the no-criteria mode to list every process.
* [](./lldbgdbremote.md#qprocessinfopid-pid-platform-extension)
* [](./lldbgdbremote.md#vfile-chmod-qplatform-chmod)
* [](./lldbgdbremote.md#qpathcomplete-platform-extension)
* [](./lldbgdbremote.md#vfile-size)
* [](./lldbgdbremote.md#vfile-mode)
* [](./lldbgdbremote.md#vfile-unlink)
* [](./lldbgdbremote.md#vfile-symlink)
* [](./lldbgdbremote.md#vfile-open)
* [](./lldbgdbremote.md#vfile-close)
* [](./lldbgdbremote.md#vfile-pread)
* [](./lldbgdbremote.md#vfile-pwrite)

The remote platform must be able to launch processes so that debugserver
can attach to them. This requires:
* [](./lldbgdbremote.md#qsetdisableaslr-bool)
* [](./lldbgdbremote.md#qsetdetachonerror)
* [](./lldbgdbremote.md#qsetstdin-ascii-hex-path-qsetstdout-ascii-hex-path-qsetstderr-ascii-hex-path) (all 3 variants)
* [](./lldbgdbremote.md#qenvironment-name-value)
* [](./lldbgdbremote.md#qenvironmenthexencoded-hex-encoding-name-value)
* [](./lldbgdbremote.md#a-launch-args-packet)
* [](./lldbgdbremote.md#qprocessinfo)
* [](./lldbgdbremote.md#qlaunchsuccess)
