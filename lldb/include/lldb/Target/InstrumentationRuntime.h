//===-- InstrumentationRuntime.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_InstrumentationRuntime_h_
#define liblldb_InstrumentationRuntime_h_

// C Includes
// C++ Includes
#include <vector>
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/lldb-types.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/StructuredData.h"

namespace lldb_private {
    
typedef std::map<lldb::InstrumentationRuntimeType, lldb::InstrumentationRuntimeSP> InstrumentationRuntimeCollection;
    
class InstrumentationRuntime :
    public std::enable_shared_from_this<InstrumentationRuntime>,
    public PluginInterface
{
public:
    
    static void
    ModulesDidLoad(lldb_private::ModuleList &module_list, Process *process, InstrumentationRuntimeCollection &runtimes);
    
    virtual void
    ModulesDidLoad(lldb_private::ModuleList &module_list);
    
    virtual bool
    IsActive();

    virtual lldb::ThreadCollectionSP
    GetBacktracesFromExtendedStopInfo(StructuredData::ObjectSP info);
    
};
    
} // namespace lldb_private

#endif  // liblldb_InstrumentationRuntime_h_
