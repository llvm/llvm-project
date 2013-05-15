//===-- ArchSpec.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ArchSpec_h_
#define liblldb_ArchSpec_h_

#if defined(__cplusplus)

#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

namespace lldb_private {

struct CoreDefinition;    

//----------------------------------------------------------------------
/// @class ArchSpec ArchSpec.h "lldb/Core/ArchSpec.h"
/// @brief An architecture specification class.
///
/// A class designed to be created from a cpu type and subtype, a
/// string representation, or an llvm::Triple.  Keeping all of the
/// conversions of strings to architecture enumeration values confined
/// to this class allows new architecture support to be added easily.
//----------------------------------------------------------------------
class ArchSpec
{
public:
    enum Core
    {
        eCore_arm_generic,
        eCore_arm_armv4,
        eCore_arm_armv4t,
        eCore_arm_armv5,
        eCore_arm_armv5e,
        eCore_arm_armv5t,
        eCore_arm_armv6,
        eCore_arm_armv7,
        eCore_arm_armv7f,
        eCore_arm_armv7s,
        eCore_arm_armv7k,
        eCore_arm_armv7m,
        eCore_arm_armv7em,
        eCore_arm_xscale,  
        eCore_thumb,
        eCore_thumbv4t,
        eCore_thumbv5,
        eCore_thumbv5e,
        eCore_thumbv6,
        eCore_thumbv7,
        eCore_thumbv7f,
        eCore_thumbv7s,
        eCore_thumbv7k,
        eCore_thumbv7m,
        eCore_thumbv7em,
        
        eCore_ppc_generic,
        eCore_ppc_ppc601,
        eCore_ppc_ppc602,
        eCore_ppc_ppc603,
        eCore_ppc_ppc603e,
        eCore_ppc_ppc603ev,
        eCore_ppc_ppc604,
        eCore_ppc_ppc604e,
        eCore_ppc_ppc620,
        eCore_ppc_ppc750,
        eCore_ppc_ppc7400,
        eCore_ppc_ppc7450,
        eCore_ppc_ppc970,
        
        eCore_ppc64_generic,
        eCore_ppc64_ppc970_64,
        
        eCore_sparc_generic,
        
        eCore_sparc9_generic,
        
        eCore_x86_32_i386,
        eCore_x86_32_i486,
        eCore_x86_32_i486sx,
        
        eCore_x86_64_x86_64,
        eCore_uknownMach32,
        eCore_uknownMach64,
        kNumCores,

        kCore_invalid,
        // The following constants are used for wildcard matching only
        kCore_any,
        kCore_arm_any,
        kCore_ppc_any,
        kCore_ppc64_any,
        kCore_x86_32_any,

        kCore_arm_first     = eCore_arm_generic,
        kCore_arm_last      = eCore_arm_xscale,

        kCore_thumb_first   = eCore_thumb,
        kCore_thumb_last    = eCore_thumbv7em,

        kCore_ppc_first     = eCore_ppc_generic,
        kCore_ppc_last      = eCore_ppc_ppc970,

        kCore_ppc64_first   = eCore_ppc64_generic,
        kCore_ppc64_last    = eCore_ppc64_ppc970_64,

        kCore_x86_32_first  = eCore_x86_32_i386,
        kCore_x86_32_last   = eCore_x86_32_i486sx
    };

    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Default constructor that initializes the object with invalid
    /// cpu type and subtype values.
    //------------------------------------------------------------------
    ArchSpec ();

    //------------------------------------------------------------------
    /// Constructor over triple.
    ///
    /// Constructs an ArchSpec with properties consistent with the given
    /// Triple.
    //------------------------------------------------------------------
    explicit 
    ArchSpec (const llvm::Triple &triple);
    explicit 
    ArchSpec (const char *triple_cstr);
    explicit 
    ArchSpec (const char *triple_cstr, Platform *platform);
    //------------------------------------------------------------------
    /// Constructor over architecture name.
    ///
    /// Constructs an ArchSpec with properties consistent with the given
    /// object type and architecture name.
    //------------------------------------------------------------------
    explicit 
    ArchSpec (ArchitectureType arch_type,
              uint32_t cpu_type,
              uint32_t cpu_subtype);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~ArchSpec ();

    //------------------------------------------------------------------
    /// Assignment operator.
    ///
    /// @param[in] rhs another ArchSpec object to copy.
    ///
    /// @return A const reference to this object.
    //------------------------------------------------------------------
    const ArchSpec&
    operator= (const ArchSpec& rhs);

    static size_t
    AutoComplete (const char *name, 
                  StringList &matches);

    //------------------------------------------------------------------
    /// Returns a static string representing the current architecture.
    ///
    /// @return A static string correcponding to the current
    ///         architecture.
    //------------------------------------------------------------------
    const char *
    GetArchitectureName () const;

    //------------------------------------------------------------------
    /// Clears the object state.
    ///
    /// Clears the object state back to a default invalid state.
    //------------------------------------------------------------------
    void
    Clear ();

    //------------------------------------------------------------------
    /// Returns the size in bytes of an address of the current
    /// architecture.
    ///
    /// @return The byte size of an address of the current architecture.
    //------------------------------------------------------------------
    uint32_t
    GetAddressByteSize () const;

    //------------------------------------------------------------------
    /// Returns a machine family for the current architecture.
    ///
    /// @return An LLVM arch type.
    //------------------------------------------------------------------
    llvm::Triple::ArchType
    GetMachine () const;

    //------------------------------------------------------------------
    /// Tests if this ArchSpec is valid.
    ///
    /// @return True if the current architecture is valid, false
    ///         otherwise.
    //------------------------------------------------------------------
    bool
    IsValid () const
    {
        return m_core >= eCore_arm_generic && m_core < kNumCores;
    }

    bool
    TripleVendorWasSpecified() const
    {
        return !m_triple.getVendorName().empty();
    }

    bool
    TripleOSWasSpecified() const
    {
        return !m_triple.getOSName().empty();
    }

    //------------------------------------------------------------------
    /// Sets this ArchSpec according to the given architecture name.
    ///
    /// The architecture name can be one of the generic system default
    /// values:
    ///
    /// @li \c LLDB_ARCH_DEFAULT - The arch the current system defaults
    ///        to when a program is launched without any extra
    ///        attributes or settings.
    /// @li \c LLDB_ARCH_DEFAULT_32BIT - The default host architecture
    ///        for 32 bit (if any).
    /// @li \c LLDB_ARCH_DEFAULT_64BIT - The default host architecture
    ///        for 64 bit (if any).
    ///
    /// Alternatively, if the object type of this ArchSpec has been
    /// configured,  a concrete architecture can be specified to set
    /// the CPU type ("x86_64" for example).
    ///
    /// Finally, an encoded object and archetecture format is accepted.
    /// The format contains an object type (like "macho" or "elf"),
    /// followed by a platform dependent encoding of CPU type and
    /// subtype.  For example:
    ///
    ///     "macho"        : Specifies an object type of MachO.
    ///     "macho-16-6"   : MachO specific encoding for ARMv6.
    ///     "elf-43        : ELF specific encoding for Sparc V9.
    ///
    /// @param[in] arch_name The name of an architecture.
    ///
    /// @return True if @p arch_name was successfully translated, false
    ///         otherwise.
    //------------------------------------------------------------------
//    bool
//    SetArchitecture (const llvm::StringRef& arch_name);
//
//    bool
//    SetArchitecture (const char *arch_name);
    
    //------------------------------------------------------------------
    /// Change the architecture object type and CPU type.
    ///
    /// @param[in] arch_type The object type of this ArchSpec.
    ///
    /// @param[in] cpu The required CPU type.
    ///
    /// @return True if the object and CPU type were sucessfully set.
    //------------------------------------------------------------------
    bool
    SetArchitecture (ArchitectureType arch_type, 
                     uint32_t cpu,
                     uint32_t sub);

    //------------------------------------------------------------------
    /// Returns the byte order for the architecture specification.
    ///
    /// @return The endian enumeration for the current endianness of
    ///     the architecture specification
    //------------------------------------------------------------------
    lldb::ByteOrder
    GetByteOrder () const;

    //------------------------------------------------------------------
    /// Sets this ArchSpec's byte order.
    ///
    /// In the common case there is no need to call this method as the
    /// byte order can almost always be determined by the architecture.
    /// However, many CPU's are bi-endian (ARM, Alpha, PowerPC, etc)
    /// and the default/assumed byte order may be incorrect.
    //------------------------------------------------------------------
    void
    SetByteOrder (lldb::ByteOrder byte_order)
    {
        m_byte_order = byte_order;
    }

    uint32_t
    GetMinimumOpcodeByteSize() const;

    uint32_t
    GetMaximumOpcodeByteSize() const;

    Core
    GetCore () const
    {
        return m_core;
    }

    uint32_t
    GetMachOCPUType () const;

    uint32_t
    GetMachOCPUSubType () const;

    //------------------------------------------------------------------
    /// Architecture tripple accessor.
    ///
    /// @return A triple describing this ArchSpec.
    //------------------------------------------------------------------
    llvm::Triple &
    GetTriple ()
    {
        return m_triple;
    }

    //------------------------------------------------------------------
    /// Architecture tripple accessor.
    ///
    /// @return A triple describing this ArchSpec.
    //------------------------------------------------------------------
    const llvm::Triple &
    GetTriple () const
    {
        return m_triple;
    }

    //------------------------------------------------------------------
    /// Architecture tripple setter.
    ///
    /// Configures this ArchSpec according to the given triple.  If the 
    /// triple has unknown components in all of the vendor, OS, and 
    /// the optional environment field (i.e. "i386-unknown-unknown")
    /// then default values are taken from the host.  Architecture and
    /// environment components are used to further resolve the CPU type
    /// and subtype, endian characteristics, etc.
    ///
    /// @return A triple describing this ArchSpec.
    //------------------------------------------------------------------
    bool
    SetTriple (const llvm::Triple &triple);

    bool
    SetTriple (const char *triple_cstr);

    bool
    SetTriple (const char *triple_cstr,
               Platform *platform);
    
    //------------------------------------------------------------------
    /// Returns the default endianness of the architecture.
    ///
    /// @return The endian enumeration for the default endianness of
    ///         the architecture.
    //------------------------------------------------------------------
    lldb::ByteOrder
    GetDefaultEndian () const;

    //------------------------------------------------------------------
    /// Compare an ArchSpec to another ArchSpec, requiring an exact cpu 
    /// type match between them.  
    /// e.g. armv7s is not an exact match with armv7 - this would return false
    ///
    /// @return true if the two ArchSpecs match.
    //------------------------------------------------------------------
    bool
    IsExactMatch (const ArchSpec& rhs) const;

    //------------------------------------------------------------------
    /// Compare an ArchSpec to another ArchSpec, requiring a compatible
    /// cpu type match between them.  
    /// e.g. armv7s is compatible with armv7 - this method would return true
    ///
    /// @return true if the two ArchSpecs are compatible
    //------------------------------------------------------------------
    bool
    IsCompatibleMatch (const ArchSpec& rhs) const;

protected:
    bool
    IsEqualTo (const ArchSpec& rhs, bool exact_match) const;

    llvm::Triple m_triple;
    Core m_core;
    lldb::ByteOrder m_byte_order;

    // Called when m_def or m_entry are changed.  Fills in all remaining
    // members with default values.
    void
    CoreUpdated (bool update_triple);
};

//------------------------------------------------------------------
/// @fn bool operator< (const ArchSpec& lhs, const ArchSpec& rhs)
/// @brief Less than operator.
///
/// Tests two ArchSpec objects to see if \a lhs is less than \a
/// rhs.
///
/// @param[in] lhs The Left Hand Side ArchSpec object to compare.
/// @param[in] rhs The Left Hand Side ArchSpec object to compare.
///
/// @return true if \a lhs is less than \a rhs
//------------------------------------------------------------------
bool operator< (const ArchSpec& lhs, const ArchSpec& rhs);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // #ifndef liblldb_ArchSpec_h_
