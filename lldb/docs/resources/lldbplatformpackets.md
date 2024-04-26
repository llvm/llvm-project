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

* [QStartNoAckMode](./lldbgdbremote.md#qstartnoackmode)
* [qHostInfo](./lldbgdbremote.md#qhostinfo)
* [qModuleInfo](./lldbgdbremote.md#qmoduleinfo-module-path-arch-triple)
* [qGetWorkingDir](./lldbgdbremote.md#qgetworkingdir)
* [QSetWorkingDir](./lldbgdbremote.md#qsetworkingdir-ascii-hex-path)
* [qPlatform_mkdir](./lldbgdbremote.md#qplatform-mkdir)
* [qPlatform_shell](./lldbgdbremote.md#qplatform-shell)
* [qLaunchGDBServer](./lldbgdbremote.md#qlaunchgdbserver-platform-extension)
* [qKillSpawnedProcess](./lldbgdbremote.md#qkillspawnedprocess-platform-extension)
* [qProcessInfoPID](./lldbgdbremote.md#qprocessinfopid-pid-platform-extension)
  * It is likely that you only need to support the `pid` and `name` fields.
* [qProcessInfo](./lldbgdbremote.md#qprocessinfo)
  * The lldb test suite currently only uses `name_match:equals` and the no-criteria mode to list every process.
* [qPathComplete](./lldbgdbremote.md#qpathcomplete-platform-extension)
* [vFile:chmod](./lldbgdbremote.md#vfile-chmod-qplatform-chmod)
* [vFile:size](./lldbgdbremote.md#vfile-size)
* [vFile:mode](./lldbgdbremote.md#vfile-mode)
* [vFile:unlink](./lldbgdbremote.md#vfile-unlink)
* [vFile:symlink](./lldbgdbremote.md#vfile-symlink)
* [vFile:open](./lldbgdbremote.md#vfile-open)
* [vFile:close](./lldbgdbremote.md#vfile-close)
* [vFile:pread](./lldbgdbremote.md#vfile-pread)
* [vFile:pwrite](./lldbgdbremote.md#vfile-pwrite)

The remote platform must be able to launch processes so that debugserver
can attach to them. This requires the following packets in addition to the
previous list:
* [QSetDisableASLR](./lldbgdbremote.md#qsetdisableaslr-bool)
* [QSetDetatchOnError](./lldbgdbremote.md#qsetdetachonerror)
* [QSetSTDIN / QSetSTDOUT / QSetSTDERR](./lldbgdbremote.md#qsetstdin-ascii-hex-path-qsetstdout-ascii-hex-path-qsetstderr-ascii-hex-path) (all 3)
* [QEnvironment](./lldbgdbremote.md#qenvironment-name-value)
* [QEnvironmentHexEncoded](./lldbgdbremote.md#qenvironmenthexencoded-hex-encoding-name-value)
* [A](./lldbgdbremote.md#a-launch-args-packet)
* [qLaunchSuccess](./lldbgdbremote.md#qlaunchsuccess)
