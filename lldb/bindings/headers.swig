/* C++ headers to be included. */
%{
#include <algorithm>
#include <string>
%}

/* The liblldb header files to be included. */
%{
#include "lldb/lldb-public.h"
#include "lldb/API/LLDB.h"
%}

%{
// helper struct to creating language bindings for SetUUID/GetUUID bytes.
// It only exists in swig generated `LLDBWrapLANG.cpp` file
struct UUIDSpan {
    const uint8_t* data;
    size_t length;
};
%}
