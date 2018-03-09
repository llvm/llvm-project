/*
   lldb.swig

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for various scripting languages, to enable them to call the
   liblldb Script Bridge functions.
*/

/* Define our module docstring. */
%define DOCSTRING
"The lldb module contains the public APIs for Python binding.

Some of the important classes are described here:

o SBTarget: Represents the target program running under the debugger.
o SBProcess: Represents the process associated with the target program.
o SBThread: Represents a thread of execution. SBProcess contains SBThread(s).
o SBFrame: Represents one of the stack frames associated with a thread. SBThread
      contains SBFrame(s).
o SBSymbolContext: A container that stores various debugger related info.
o SBValue: Represents the value of a variable, a register, or an expression.
o SBModule: Represents an executable image and its associated object and symbol
      files.  SBTarget contains SBModule(s).
o SBBreakpoint: Represents a logical breakpoint and its associated settings.
      SBTarget contains SBBreakpoint(s).
o SBSymbol: Represents the symbol possibly associated with a stack frame.
o SBCompileUnit: Represents a compilation unit, or compiled source file.
o SBFunction: Represents a generic function, which can be inlined or not.
o SBBlock: Represents a lexical block. SBFunction contains SBBlock(s).
o SBLineEntry: Specifies an association with a contiguous range of instructions
      and a source file location. SBCompileUnit contains SBLineEntry(s)."
%enddef

/*
Since version 3.0.9, swig's logic for importing the native module has changed in
a way that is incompatible with our usage of the python module as __init__.py
(See swig bug #769).  Fortunately, since version 3.0.11, swig provides a way for
us to override the module import logic to suit our needs. This does that.

Older swig versions will simply ignore this setting.
*/
%define MODULEIMPORT
"try:
    # Try a relative import first
    from . import $module
except ImportError:
    # Maybe absolute import will work (if we're being loaded from lldb, it
    # should).
    import $module"
%enddef
// These versions will not generate working python modules, so error out early.
#if SWIG_VERSION >= 0x030009 && SWIG_VERSION < 0x030011
#error Swig versions 3.0.9 and 3.0.10 are incompatible with lldb.
#endif

// The name of the module to be created.
%module(docstring=DOCSTRING, moduleimport=MODULEIMPORT) lldb

// Parameter types will be used in the autodoc string.
%feature("autodoc", "1");

%pythoncode%{
import uuid
import re
import os

import six
%}
%include "./Python/python-typemaps.swig"

/* C++ headers to be included. */
%{
#include <algorithm>
#include <string>
%}

/* The liblldb header files to be included. */
%{
#include "lldb/lldb-public.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBAttachInfo.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBBreakpointName.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBData.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBExecutionContext.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBLanguageRuntime.h"
#include "lldb/API/SBLaunchInfo.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "lldb/API/SBMemoryRegionInfoList.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/API/SBPlatform.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBProcessInfo.h"
#include "lldb/API/SBQueue.h"
#include "lldb/API/SBQueueItem.h"
#include "lldb/API/SBSection.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBThreadPlan.h"
#include "lldb/API/SBTrace.h"
#include "lldb/API/SBTraceOptions.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBTypeCategory.h"
#include "lldb/API/SBTypeEnumMember.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "lldb/API/SBVariablesOptions.h"
#include "lldb/API/SBWatchpoint.h"
#include "lldb/API/SBUnixSignals.h"

#include "../source/Plugins/ScriptInterpreter/Python/PythonDataObjects.h"

#include "../scripts/Python/python-swigsafecast.swig"
%}

/* Various liblldb typedefs that SWIG needs to know about.  */
#define __extension__ /* Undefine GCC keyword to make Swig happy when processing glibc's stdint.h. */
/* The ISO C99 standard specifies that in C++ implementations limit macros such
   as INT32_MAX should only be defined if __STDC_LIMIT_MACROS is. */
#define __STDC_LIMIT_MACROS
%include "stdint.i"

%include "lldb/lldb-defines.h"
%include "lldb/lldb-enumerations.h"
%include "lldb/lldb-forward.h"
%include "lldb/lldb-types.h"

/* Forward declaration of SB classes. */
%include "lldb/API/SBDefines.h"

/* Python interface files with docstrings. */
%include "./interface/SBAddress.i"
%include "./interface/SBAttachInfo.i"
%include "./interface/SBBlock.i"
%include "./interface/SBBreakpoint.i"
%include "./interface/SBBreakpointLocation.i"
%include "./interface/SBBreakpointName.i"
%include "./interface/SBBroadcaster.i"
%include "./interface/SBCommandInterpreter.i"
%include "./interface/SBCommandReturnObject.i"
%include "./interface/SBCommunication.i"
%include "./interface/SBCompileUnit.i"
%include "./interface/SBData.i"
%include "./interface/SBDebugger.i"
%include "./interface/SBDeclaration.i"
%include "./interface/SBError.i"
%include "./interface/SBEvent.i"
%include "./interface/SBExecutionContext.i"
%include "./interface/SBExpressionOptions.i"
%include "./interface/SBFileSpec.i"
%include "./interface/SBFileSpecList.i"
%include "./interface/SBFrame.i"
%include "./interface/SBFunction.i"
%include "./interface/SBHostOS.i"
%include "./interface/SBInstruction.i"
%include "./interface/SBInstructionList.i"
%include "./interface/SBLanguageRuntime.i"
%include "./interface/SBLaunchInfo.i"
%include "./interface/SBLineEntry.i"
%include "./interface/SBListener.i"
%include "./interface/SBMemoryRegionInfo.i"
%include "./interface/SBMemoryRegionInfoList.i"
%include "./interface/SBModule.i"
%include "./interface/SBModuleSpec.i"
%include "./interface/SBPlatform.i"
%include "./interface/SBProcess.i"
%include "./interface/SBProcessInfo.i"
%include "./interface/SBQueue.i"
%include "./interface/SBQueueItem.i"
%include "./interface/SBSection.i"
%include "./interface/SBSourceManager.i"
%include "./interface/SBStream.i"
%include "./interface/SBStringList.i"
%include "./interface/SBStructuredData.i"
%include "./interface/SBSymbol.i"
%include "./interface/SBSymbolContext.i"
%include "./interface/SBSymbolContextList.i"
%include "./interface/SBTarget.i"
%include "./interface/SBThread.i"
%include "./interface/SBThreadCollection.i"
%include "./interface/SBThreadPlan.i"
%include "./interface/SBTrace.i"
%include "./interface/SBTraceOptions.i"
%include "./interface/SBType.i"
%include "./interface/SBTypeCategory.i"
%include "./interface/SBTypeEnumMember.i"
%include "./interface/SBTypeFilter.i"
%include "./interface/SBTypeFormat.i"
%include "./interface/SBTypeNameSpecifier.i"
%include "./interface/SBTypeSummary.i"
%include "./interface/SBTypeSynthetic.i"
%include "./interface/SBValue.i"
%include "./interface/SBValueList.i"
%include "./interface/SBVariablesOptions.i"
%include "./interface/SBWatchpoint.i"
%include "./interface/SBUnixSignals.i"

%include "./Python/python-extensions.swig"

%include "./Python/python-wrapper.swig"
