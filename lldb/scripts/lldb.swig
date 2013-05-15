/*
   lldb.swig

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for various scripting languages, to enable them to call the
   liblldb Script Bridge functions.
*/

/* Define our module docstring. */
%define DOCSTRING
"The lldb module contains the public APIs for Python binding.

Some of the important classes are describe here:

o SBTarget: Represents the target program running under the debugger.
o SBProcess: Represents the process associated with the target program.
o SBThread: Represents a thread of execution. SBProcess contains SBThread(s).
o SBFrame: Represents one of the stack frames associated with a thread. SBThread
      contains SBFrame(s).
o SBSymbolContext: A container that stores various debugger related info.
o SBValue: Represents the value of a variable, a register, or an expression.
o SBModule: Represents an executable image and its associated object and symbol
      files.  SBTarget conatins SBModule(s).
o SBBreakpoint: Represents a logical breakpoint and its associated settings.
      SBTarget conatins SBBreakpoint(s).
o SBSymbol: Represents the symbol possibly associated with a stack frame.
o SBCompileUnit: Represents a compilation unit, or compiled source file.
o SBFunction: Represents a generic function, which can be inlined or not.
o SBBlock: Represents a lexical block. SBFunction contains SBBlock(s).
o SBLineEntry: Specifies an association with a contiguous range of instructions
      and a source file location. SBCompileUnit contains SBLineEntry(s)."
%enddef

// The name of the module to be created.
%module(docstring=DOCSTRING) lldb

// Parameter types will be used in the autodoc string.
%feature("autodoc", "1");

%pythoncode%{
import uuid
import re
import os
%}
%include "./Python/python-typemaps.swig"

/* C++ headers to be included. */
%{
#include <string>
%}

/* The liblldb header files to be included. */
%{
#include "lldb/lldb-public.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
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
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSection.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBTypeCategory.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "lldb/API/SBWatchpoint.h"
%}

/* Various liblldb typedefs that SWIG needs to know about.  */
#define __extension__ /* Undefine GCC keyword to make Swig happy when processing glibc's stdint.h. */
%include "stdint.i"
%include "lldb/lldb-defines.h"
%include "lldb/lldb-enumerations.h"
%include "lldb/lldb-forward.h"
%include "lldb/lldb-types.h"

/* Forward declaration of SB classes. */
%include "lldb/API/SBDefines.h"

/* Python interface files with docstrings. */
%include "./Python/interface/SBAddress.i"
%include "./Python/interface/SBBlock.i"
%include "./Python/interface/SBBreakpoint.i"
%include "./Python/interface/SBBreakpointLocation.i"
%include "./Python/interface/SBBroadcaster.i"
%include "./Python/interface/SBCommandInterpreter.i"
%include "./Python/interface/SBCommandReturnObject.i"
%include "./Python/interface/SBCommunication.i"
%include "./Python/interface/SBCompileUnit.i"
%include "./Python/interface/SBData.i"
%include "./Python/interface/SBDebugger.i"
%include "./Python/interface/SBDeclaration.i"
%include "./Python/interface/SBError.i"
%include "./Python/interface/SBEvent.i"
%include "./Python/interface/SBExpressionOptions.i"
%include "./Python/interface/SBFileSpec.i"
%include "./Python/interface/SBFileSpecList.i"
%include "./Python/interface/SBFrame.i"
%include "./Python/interface/SBFunction.i"
%include "./Python/interface/SBHostOS.i"
%include "./Python/interface/SBInputReader.i"
%include "./Python/interface/SBInstruction.i"
%include "./Python/interface/SBInstructionList.i"
%include "./Python/interface/SBLineEntry.i"
%include "./Python/interface/SBListener.i"
%include "./Python/interface/SBModule.i"
%include "./Python/interface/SBProcess.i"
%include "./Python/interface/SBSection.i"
%include "./Python/interface/SBSourceManager.i"
%include "./Python/interface/SBStream.i"
%include "./Python/interface/SBStringList.i"
%include "./Python/interface/SBSymbol.i"
%include "./Python/interface/SBSymbolContext.i"
%include "./Python/interface/SBSymbolContextList.i"
%include "./Python/interface/SBTarget.i"
%include "./Python/interface/SBThread.i"
%include "./Python/interface/SBType.i"
%include "./Python/interface/SBTypeCategory.i"
%include "./Python/interface/SBTypeFilter.i"
%include "./Python/interface/SBTypeFormat.i"
%include "./Python/interface/SBTypeNameSpecifier.i"
%include "./Python/interface/SBTypeSummary.i"
%include "./Python/interface/SBTypeSynthetic.i"
%include "./Python/interface/SBValue.i"
%include "./Python/interface/SBValueList.i"
%include "./Python/interface/SBWatchpoint.i"

%include "./Python/python-extensions.swig"

%include "./Python/python-wrapper.swig"
