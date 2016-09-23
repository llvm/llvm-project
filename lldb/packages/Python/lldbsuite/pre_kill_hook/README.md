# pre\_kill\_hook package

## Overview

The pre\_kill\_hook package provides a per-platform method for running code
after a test process times out but before the concurrent test runner kills the
timed-out process.

## Detailed Description of Usage

If a platform defines the hook, then the hook gets called right after a timeout
is detected in a test run, but before the process is killed.

The pre-kill-hook mechanism works as follows:

* When a timeout is detected in the process_control.ProcessDriver class that
  runs the per-test lldb process, a new overridable on\_timeout\_pre\_kill() method
  is called on the ProcessDriver instance.

* The concurrent test driver's derived ProcessDriver overrides this method. It
  looks to see if a module called
  "lldbsuite.pre\_kill\_hook.{platform-system-name}" module exists, where
  platform-system-name is replaced with platform.system().lower().  (e.g.
  "Darwin" becomes the darwin.py module).
  
  * If that module doesn't exist, the rest of the new behavior is skipped.

  * If that module does exist, it is loaded, and the method
    "do\_pre\_kill(process\_id, context\_dict, output\_stream)" is called. If
    that method throws an exception, we log it and we ignore further processing
    of the pre-killed process.

  * The process\_id argument of the do\_pre\_kill function is the process id as
    returned by the ProcessDriver.pid property.
  
  * The output\_stream argument of the do\_pre\_kill function takes a file-like
    object. Output to be collected from doing any processing on the
    process-to-be-killed should be written into the file-like object. The
    current impl uses a six.StringIO and then writes this output to
    {TestFilename}-{pid}.sample in the session directory.
    
* Platforms where platform.system() is "Darwin" will get a pre-kill action that
  runs the 'sample' program on the lldb that has timed out. That data will be
  collected on CI and analyzed to determine what is happening during timeouts.
  (This has an advantage over a core in that it is much smaller and that it
  clearly demonstrates any liveness of the process, if there is any).

## Running the tests

To run the tests in the pre\_kill\_hook package, open a console, change into
this directory and run the following:

```
python -m unittest discover
```
