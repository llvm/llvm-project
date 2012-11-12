//===-- SBSourceManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBStream.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"

#include "lldb/Target/Target.h"

namespace lldb_private
{
    class SourceManagerImpl
    {
    public:
        SourceManagerImpl (const lldb::DebuggerSP &debugger_sp)
        {
            m_debugger_sp = debugger_sp;
        }
        
        SourceManagerImpl (const lldb::TargetSP &target_sp)
        {
            m_target_sp = target_sp;
        }
        
        SourceManagerImpl (const SourceManagerImpl &rhs)
        {
            if (&rhs == this)
                return;
            m_debugger_sp = rhs.m_debugger_sp;
            m_target_sp   = rhs.m_target_sp;
        }
        
        size_t
        DisplaySourceLinesWithLineNumbers (const lldb_private::FileSpec &file,
                                           uint32_t line,
                                           uint32_t context_before,
                                           uint32_t context_after,
                                           const char *current_line_cstr,
                                           lldb_private::Stream *s)
        {
            if (!file)
                return 0;
            
            if (m_debugger_sp)
                return m_debugger_sp->GetSourceManager().DisplaySourceLinesWithLineNumbers (file,
                                                                                            line,
                                                                                            context_before,
                                                                                            context_after,
                                                                                            current_line_cstr,
                                                                                            s);
            else if (m_target_sp)
                return m_target_sp->GetSourceManager().DisplaySourceLinesWithLineNumbers (file,
                                                                                          line,
                                                                                          context_before,
                                                                                          context_after,
                                                                                          current_line_cstr,
                                                                                          s);
            else
                return 0;
        }
        
    private:
        lldb::DebuggerSP m_debugger_sp;
        lldb::TargetSP   m_target_sp;
        
    };
}

using namespace lldb;
using namespace lldb_private;

SBSourceManager::SBSourceManager (const SBDebugger &debugger)
{
    m_opaque_ap.reset(new SourceManagerImpl (debugger.get_sp()));
}

SBSourceManager::SBSourceManager (const SBTarget &target)
{
    m_opaque_ap.reset(new SourceManagerImpl (target.GetSP()));
}

SBSourceManager::SBSourceManager (const SBSourceManager &rhs)
{
    if (&rhs == this)
        return;
        
    m_opaque_ap.reset(new SourceManagerImpl (*(rhs.m_opaque_ap.get())));
}

const lldb::SBSourceManager &
SBSourceManager::operator = (const lldb::SBSourceManager &rhs)
{
    m_opaque_ap.reset (new SourceManagerImpl (*(rhs.m_opaque_ap.get())));
    return *this;
}

SBSourceManager::~SBSourceManager()
{
}

size_t
SBSourceManager::DisplaySourceLinesWithLineNumbers
(
    const SBFileSpec &file,
    uint32_t line,
    uint32_t context_before,
    uint32_t context_after,
    const char *current_line_cstr,
    SBStream &s
)
{
    if (m_opaque_ap.get() == NULL)
        return 0;

    return m_opaque_ap->DisplaySourceLinesWithLineNumbers (file.ref(),
                                                           line,
                                                           context_before,
                                                           context_after,
                                                           current_line_cstr,
                                                           s.get());
}
