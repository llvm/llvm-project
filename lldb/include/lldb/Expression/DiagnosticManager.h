//===-- DiagnosticManager.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_DiagnosticManager_h
#define lldb_DiagnosticManager_h

#include "lldb/lldb-types.h"

#include <string>
#include <vector>

namespace lldb_private
{

enum DiagnosticOrigin
{
    eDiagnosticOriginUnknown = 0,
    eDiagnosticOriginLLDB,
    eDiagnosticOriginClang,
    eDiagnosticOriginGo,
    eDiagnosticOriginSwift,
    eDiagnosticOriginLLVM
};

enum DiagnosticSeverity
{
    eDiagnosticSeverityError,
    eDiagnosticSeverityWarning,
    eDiagnosticSeverityRemark
};

const uint32_t LLDB_INVALID_COMPILER_ID = UINT32_MAX;

struct Diagnostic
{
    std::string message;
    uint32_t compiler_id; // Compiler-specific diagnostic ID
    DiagnosticSeverity severity;
    DiagnosticOrigin origin;
};

typedef std::vector<Diagnostic> DiagnosticList;

class DiagnosticManager
{
public:
    void
    Clear()
    {
        m_diagnostics.clear();
    }

    const DiagnosticList &
    Diagnostics()
    {
        return m_diagnostics;
    }

    void
    AddDiagnostic(const char *message, DiagnosticSeverity severity, DiagnosticOrigin origin,
                  uint32_t compiler_id = LLDB_INVALID_COMPILER_ID)
    {
        m_diagnostics.push_back({std::string(message), compiler_id, severity, origin});
    }

    size_t
    Printf(DiagnosticSeverity severity, const char *format, ...) __attribute__((format(printf, 3, 4)));
    size_t
    PutCString(DiagnosticSeverity severity, const char *cstr);

    void
    AppendMessageToDiagnostic(const char *cstr)
    {
        if (m_diagnostics.size())
        {
            m_diagnostics.back().message.push_back('\n');
            m_diagnostics.back().message.append(cstr);
        }
    }

    // Returns a string containing errors in this format:
    //
    // "error: error text\n
    // warning: warning text\n
    // remark text\n"
    std::string
    GetString(char separator = '\n');

    void
    Dump(Log *log);

private:
    DiagnosticList m_diagnostics;
};
}

#endif /* lldb_DiagnosticManager_h */
