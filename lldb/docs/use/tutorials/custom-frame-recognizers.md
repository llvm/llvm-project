# Detecting Patterns With Recognizers

Frame recognizers allow for retrieving information about special frames based
on ABI, arguments or other special properties of that frame, even without
source code or debug info. Currently, one use case is to extract function
arguments that would otherwise be inaccessible, or augment existing arguments.

Adding a custom frame recognizer is done by implementing a Python class and
using the `frame recognizer add` command. The Python class should implement the
`get_recognized_arguments` method and it will receive an argument of type
`lldb.SBFrame` representing the current frame that we are trying to recognize.
The method should return a (possibly empty) list of `lldb.SBValue` objects that
represent the recognized arguments.

An example of a recognizer that retrieves the file descriptor values from libc
functions 'read', 'write' and 'close' follows:

```python3
class LibcFdRecognizer:
  def get_recognized_arguments(self, frame: lldb.SBFrame):
    if frame.name in ["read", "write", "close"]:
      fd = frame.EvaluateExpression("$arg1").unsigned
      target = frame.thread.process.target
      value = target.CreateValueFromExpression("fd", "(int)%d" % fd)
      return [value]
    return []
```

The file containing this implementation can be imported via `command script import`
and then we can register this recognizer with `frame recognizer add`.

It's important to restrict the recognizer to the libc library (which is
`libsystem_kernel.dylib` on macOS) to avoid matching functions with the same name
in other modules:

```c++
(lldb) command script import .../fd_recognizer.py
(lldb) frame recognizer add -l fd_recognizer.LibcFdRecognizer -n read -s libsystem_kernel.dylib
```

When the program is stopped at the beginning of the 'read' function in libc, we can view the recognizer arguments in 'frame variable':

```c++
(lldb) b read
(lldb) r
Process 1234 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.3
    frame #0: 0x00007fff06013ca0 libsystem_kernel.dylib`read
(lldb) frame variable
(int) fd = 3
```