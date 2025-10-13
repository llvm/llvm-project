Python Reference
================

The entire LLDB API is available as Python functions through a script bridging
interface. This means the LLDB API's can be used directly from python either
interactively or to build python apps that provide debugger features.

Additionally, Python can be used as a programmatic interface within the lldb
command interpreter (we refer to this for brevity as the embedded interpreter).
Of course, in this context it has full access to the LLDB API - with some
additional conveniences we will call out in the FAQ.

Python Tutorials
-----------------

The following tutorials and documentation demonstrate various Python capabilities within LLDB:

.. toctree::
   :maxdepth: 1

   tutorials/accessing-documentation
   tutorials/python-embedded-interpreter
   tutorials/script-driven-debugging
   tutorials/breakpoint-triggered-scripts
   tutorials/creating-custom-breakpoints
   tutorials/automating-stepping-logic
   tutorials/writing-custom-commands
   tutorials/implementing-standalone-scripts
   tutorials/custom-frame-recognizers
   tutorials/extending-target-stop-hooks
