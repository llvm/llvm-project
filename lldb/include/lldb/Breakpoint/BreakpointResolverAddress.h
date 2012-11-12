//===-- BreakpointResolverAddress.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointResolverAddress_h_
#define liblldb_BreakpointResolverAddress_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointResolver.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointResolverAddress BreakpointResolverAddress.h "lldb/Breakpoint/BreakpointResolverAddress.h"
/// @brief This class sets breakpoints on a given Address.  This breakpoint only takes
/// once, and then it won't attempt to reset itself.
//----------------------------------------------------------------------

class BreakpointResolverAddress:
    public BreakpointResolver
{
public:
    BreakpointResolverAddress (Breakpoint *bkpt,
                       const Address &addr);

    virtual
    ~BreakpointResolverAddress ();

    virtual void
    ResolveBreakpoint (SearchFilter &filter);

    virtual void
    ResolveBreakpointInModules (SearchFilter &filter,
                                ModuleList &modules);

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
    static inline bool classof(const BreakpointResolverAddress *) { return true; }
    static inline bool classof(const BreakpointResolver *V) {
        return V->getResolverID() == BreakpointResolver::AddressResolver;
    }

protected:
    Address m_addr;

private:
    DISALLOW_COPY_AND_ASSIGN(BreakpointResolverAddress);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointResolverAddress_h_
