//===-- SBDefines.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBDefines_h_
#define LLDB_SBDefines_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

// Forward Declarations

namespace lldb {

class SBAddress;
class SBBlock;
class SBBreakpoint;
class SBBreakpointLocation;
class SBBroadcaster;
class SBCommand;
class SBCommandInterpreter;
class SBCommandPluginInterface;
class SBCommandReturnObject;
class SBCommunication;
class SBCompileUnit;
class SBData;
class SBDebugger;
class SBDeclaration;
class SBError;
class SBEvent;
class SBEventList;
class SBExpressionOptions;
class SBFileSpec;
class SBFileSpecList;
class SBFrame;
class SBFunction;
class SBHostOS;
class SBInputReader;
class SBInstruction;
class SBInstructionList;
class SBLineEntry;
class SBListener;
class SBModule;
class SBProcess;
class SBSourceManager;
class SBStream;
class SBStringList;
class SBSymbol;
class SBSymbolContext;
class SBSymbolContextList;
class SBTarget;
class SBThread;
class SBType;
class SBTypeCategory;
class SBTypeFilter;
class SBTypeFormat;
class SBTypeNameSpecifier;
class SBTypeSummary;
#ifndef LLDB_DISABLE_PYTHON
class SBTypeSynthetic;    
#endif
class SBTypeList;
class SBValue;
class SBValueList;
class SBWatchpoint;

}

#endif  // LLDB_SBDefines_h_
