/*
   lldb.swig

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for JavaScript (V8/Node.js), to enable them to call the
   liblldb Script Bridge functions.
*/

%module lldb

%include <std_string.i>
%include "javascript-typemaps.swig"
%include "macros.swig"
%include "headers.swig"

%{
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "../bindings/javascript/javascript-swigsafecast.swig"
#include "../source/Plugins/ScriptInterpreter/JavaScript/SWIGJavaScriptBridge.h"

// required headers for typemaps
#include "lldb/Host/File.h"

using namespace lldb_private;
using namespace lldb;
%}

%include "interfaces.swig"
%include "javascript-wrapper.swig"
