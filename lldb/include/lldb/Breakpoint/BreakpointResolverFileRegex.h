//===-- BreakpointResolverFileRegex.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointResolverFileRegex_h_
#define liblldb_BreakpointResolverFileRegex_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointResolver.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointResolverFileRegex BreakpointResolverFileRegex.h "lldb/Breakpoint/BreakpointResolverFileRegex.h"
/// @brief This class sets breakpoints by file and line.  Optionally, it will look for inlined
/// instances of the file and line specification.
//----------------------------------------------------------------------

class BreakpointResolverFileRegex :
    public BreakpointResolver
{
public:
    BreakpointResolverFileRegex (Breakpoint *bkpt,
                                RegularExpression &regex);

    virtual
    ~BreakpointResolverFileRegex ();

    virtual Searcher::CallbackReturn
    SearchCallback (SearchFilter &filter,
                    SymbolContext &context,
                    Address *addr,
                    bool containing);

    virtual Searcher::Depth
    GetDepth ();

    virtual void
    GetDescription (Stream *s);

    virtual void
    Dump (Stream *s) const;

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const BreakpointResolverFileRegex *) { return true; }
    static inline bool classof(const BreakpointResolver *V) {
        return V->getResolverID() == BreakpointResolver::FileRegexResolver;
    }

protected:
    friend class Breakpoint;
    RegularExpression m_regex; // This is the line expression that we are looking for.

private:
    DISALLOW_COPY_AND_ASSIGN(BreakpointResolverFileRegex);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointResolverFileRegex_h_
