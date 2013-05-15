//===-- BreakpointResolver.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointResolver_h_
#define liblldb_BreakpointResolver_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/ConstString.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointResolver BreakpointResolver.h "lldb/Breakpoint/BreakpointResolver.h"
/// @brief This class works with SearchFilter to resolve logical breakpoints to their
/// of concrete breakpoint locations.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
/// General Outline:
/// The BreakpointResolver is a Searcher.  In that protocol,
/// the SearchFilter asks the question "At what depth of the symbol context
/// descent do you want your callback to get called?" of the filter.  The resolver
/// answers this question (in the GetDepth method) and provides the resolution callback.
/// Each Breakpoint has a BreakpointResolver, and it calls either ResolveBreakpoint
/// or ResolveBreakpointInModules to tell it to look for new breakpoint locations.
//----------------------------------------------------------------------

class BreakpointResolver :
   public Searcher
{
public:
    //------------------------------------------------------------------
    /// The breakpoint resolver need to have a breakpoint for "ResolveBreakpoint
    /// to make sense.  It can be constructed without a breakpoint, but you have to
    /// call SetBreakpoint before ResolveBreakpoint.
    ///
    /// @param[in] bkpt
    ///   The breakpoint that owns this resolver.
    /// @param[in] resolverType
    ///   The concrete breakpoint resolver type for this breakpoint.
    ///
    /// @result
    ///   Returns breakpoint location id.
    //------------------------------------------------------------------
    BreakpointResolver (Breakpoint *bkpt, unsigned char resolverType);

    //------------------------------------------------------------------
    /// The Destructor is virtual, all significant breakpoint resolvers derive
    /// from this class.
    //------------------------------------------------------------------
    virtual
    ~BreakpointResolver ();

    //------------------------------------------------------------------
    /// This sets the breakpoint for this resolver.
    ///
    /// @param[in] bkpt
    ///   The breakpoint that owns this resolver.
    //------------------------------------------------------------------
    void
    SetBreakpoint (Breakpoint *bkpt);

    //------------------------------------------------------------------
    /// In response to this method the resolver scans all the modules in the breakpoint's
    /// target, and adds any new locations it finds.
    ///
    /// @param[in] filter
    ///   The filter that will manage the search for this resolver.
    //------------------------------------------------------------------
    virtual void
    ResolveBreakpoint (SearchFilter &filter);

    //------------------------------------------------------------------
    /// In response to this method the resolver scans the modules in the module list
    /// \a modules, and adds any new locations it finds.
    ///
    /// @param[in] filter
    ///   The filter that will manage the search for this resolver.
    //------------------------------------------------------------------
    virtual void
    ResolveBreakpointInModules (SearchFilter &filter,
                                ModuleList &modules);

    //------------------------------------------------------------------
    /// Prints a canonical description for the breakpoint to the stream \a s.
    ///
    /// @param[in] s
    ///   Stream to which the output is copied.
    //------------------------------------------------------------------
    virtual void
    GetDescription (Stream *s) = 0;

    //------------------------------------------------------------------
    /// Standard "Dump" method.  At present it does nothing.
    //------------------------------------------------------------------
    virtual void
    Dump (Stream *s) const = 0;

    //------------------------------------------------------------------
    /// An enumeration for keeping track of the concrete subclass that
    /// is actually instantiated. Values of this enumeration are kept in the 
    /// BreakpointResolver's SubclassID field. They are used for concrete type
    /// identification.
    enum ResolverTy {
        FileLineResolver, // This is an instance of BreakpointResolverFileLine
        AddressResolver,  // This is an instance of BreakpointResolverAddress
        NameResolver,      // This is an instance of BreakpointResolverName
        FileRegexResolver,
        ExceptionResolver
    };

    //------------------------------------------------------------------
    /// getResolverID - Return an ID for the concrete type of this object.  This
    /// is used to implement the LLVM classof checks.  This should not be used
    /// for any other purpose, as the values may change as LLDB evolves.
    unsigned getResolverID() const {
        return SubclassID;
    }

protected:
    Breakpoint *m_breakpoint;  // This is the breakpoint we add locations to.

private:
    // Subclass identifier (for llvm isa/dyn_cast)
    const unsigned char SubclassID;
    DISALLOW_COPY_AND_ASSIGN(BreakpointResolver);
};

} // namespace lldb_private

#endif  // liblldb_BreakpointResolver_h_
