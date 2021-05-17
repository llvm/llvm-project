Scripting Bridge API
====================

The SB APIs constitute the stable C++ API that lldb presents to external
clients, and which get processed by SWIG to produce the Python bindings to
lldb. As such it is important that they not suffer from the binary
incompatibilities that C++ is so susceptible to. We've established a few rules
to ensure that this happens.

The classes in the SB API's are all called SB<SomeName>, where SomeName is in
CamelCase starting with an upper case letter. The method names are all
CamelCase with initial capital letter as well.

All the SB API classes are non-virtual, single inheritance classes. They should
only include SBDefines.h or other SB headers as needed. There should be no
inlined method implementations in the header files, they should all be in the
implementation files. And there should be no direct ivar access.

You also need to choose the ivars for the class with care, since you can't add
or remove ivars without breaking binary compatibility. In some cases, the SB
class is a thin wrapper around an internal lldb_private object. In that case,
the class can have a single ivar, which is either a pointer, shared_ptr or
unique_ptr to the object in the lldb_private API. All the lldb_private classes
that get used this way are declared as opaque classes in lldb_forward.h, which
is included in SBDefines.h. So if you need an SB class to wrap an lldb_private
class that isn't in lldb_forward.h, add it there rather than making a direct
opaque declaration in the SB classes .h file.

If the SB Class needs some state of its own, as well as the backing object,
don't include that as a direct ivar in the SB Class. Instead, make an Impl
class in the SB's .cpp file, and then make the SB object hold a shared or
unique pointer to the Impl object. The theory behind this is that if you need
more state in the SB object, those needs are likely to change over time, and
this way the Impl class can pick up members without changing the size of the
object. An example of this is the SBValue class. Please note that you should
not put this Impl class in the lldb namespace. Failure to do so leads to
leakage of weak-linked symbols in the SBAPI.

In order to fit into the Python API's, we need to be able to default construct
all the SB objects. Since the ivars of the classes are all pointers of one sort
or other, this can easily be done, but it means all the methods must be
prepared to handle their opaque implementation pointer being empty, and doing
something reasonable. We also always have an "IsValid" method on all the SB
classes to report whether the object is empty or not.

Another piece of the SB API infrastructure is the Python (or other script
interpreter) customization. SWIG allows you to add property access, iterators
and documentation to classes, but to do that you have to use a Swig interface
file in place of the .h file. Those files have a different format than a
straight C++ header file. These files are called SB<ClassName>.i, and live in
"scripts/interface". They are constructed by starting with the associated .h
file, and adding documentation and the Python decorations, etc. We do this in a
decidedly low-tech way, by maintaining the two files in parallel. That
simplifies the build process, but it does mean that if you add a method to the
C++ API's for an SB class, you have to copy the interface to the .i file.

API Instrumentation
-------------------

The reproducer infrastructure requires API methods to be instrumented so that
they can be captured and replayed. Instrumentation consists of two macros,
``LLDB_REGISTER`` and ``LLDB_RECORD``. Both can be automatically generated with
the ``lldb-instr`` utility.

To add instrumentation for a given file, pass it to the ``lldb-instr`` tool.
Like other clang-based tools it requires a compilation database
(``compile_commands.json``) to be present in the current working directory.

::

    ./bin/lldb-instr /path/to/lldb/source/API/SBDebugger.cpp


The tool will automatically insert ``LLDB_RECORD`` macros inline, however you
will need to run ``clang-format`` over the processed file, as the tool
(intentionally) makes no attempt to get that right.

The ``LLDB_REGISTER`` macros are printed to standard out between curly braces.
You'll have to copy-paste those into the corresponding ``RegisterMethods``
function in the implementation file. This function is fully specialized in the
corresponding type.

::

  template <> void RegisterMethods<SBDebugger>(Registry &R) {
    ...
  }


When adding a new class, you'll also have to add a call to ``RegisterMethods``
in the ``SBRegistry`` constructor.

The tool can be used incrementally. However, it will ignore existing macros
even if their signature is wrong. It will only generate a ``LLDB_REGISTER`` if
it emitted a corresponding ``LLDB_RECORD`` macro.
