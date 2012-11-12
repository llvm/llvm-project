//===-- ArchSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ArchSpec.h"

#include <stdio.h>
#include <errno.h>

#include <string>

#include "llvm/Support/ELF.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

#define ARCH_SPEC_SEPARATOR_CHAR '-'


static bool cores_match (const ArchSpec::Core core1, const ArchSpec::Core core2, bool try_inverse, bool enforce_exact_match);

namespace lldb_private {

    struct CoreDefinition
    {
        ByteOrder default_byte_order;
        uint32_t addr_byte_size;
        uint32_t min_opcode_byte_size;
        uint32_t max_opcode_byte_size;
        llvm::Triple::ArchType machine;
        ArchSpec::Core core;
        const char *name;
    };

}

// This core information can be looked using the ArchSpec::Core as the index
static const CoreDefinition g_core_definitions[ArchSpec::kNumCores] =
{
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_generic     , "arm"       },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv4       , "armv4"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv4t      , "armv4t"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5       , "armv5"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5e      , "armv5e"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5t      , "armv5t"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv6       , "armv6"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7       , "armv7"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7f      , "armv7f"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7k      , "armv7k"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7s      , "armv7s"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_xscale      , "xscale"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumb           , "thumb"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv4t        , "thumbv4t"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv5         , "thumbv5"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv5e        , "thumbv5e"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv6         , "thumbv6"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7         , "thumbv7"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7f        , "thumbv7f"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7k        , "thumbv7k"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7s        , "thumbv7s"  },
    
    
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_generic     , "ppc"       },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc601      , "ppc601"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc602      , "ppc602"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603      , "ppc603"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603e     , "ppc603e"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603ev    , "ppc603ev"  },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc604      , "ppc604"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc604e     , "ppc604e"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc620      , "ppc620"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc750      , "ppc750"    },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc7400     , "ppc7400"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc7450     , "ppc7450"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc970      , "ppc970"    },
    
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::ppc64  , ArchSpec::eCore_ppc64_generic   , "ppc64"     },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::ppc64  , ArchSpec::eCore_ppc64_ppc970_64 , "ppc970-64" },
    
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::sparc  , ArchSpec::eCore_sparc_generic   , "sparc"     },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::sparcv9, ArchSpec::eCore_sparc9_generic  , "sparcv9"   },

    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i386    , "i386"      },
    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i486    , "i486"      },
    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i486sx  , "i486sx"    },

    { eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64 , ArchSpec::eCore_x86_64_x86_64  , "x86_64"    },
    { eByteOrderLittle, 4, 4, 4 , llvm::Triple::UnknownArch , ArchSpec::eCore_uknownMach32  , "unknown-mach-32" },
    { eByteOrderLittle, 8, 4, 4 , llvm::Triple::UnknownArch , ArchSpec::eCore_uknownMach64  , "unknown-mach-64" }
};

struct ArchDefinitionEntry
{
    ArchSpec::Core core;
    uint32_t cpu;
    uint32_t sub;
    uint32_t cpu_mask;
    uint32_t sub_mask;
};

struct ArchDefinition
{
    ArchitectureType type;
    size_t num_entries;
    const ArchDefinitionEntry *entries;
    const char *name;
};


uint32_t
ArchSpec::AutoComplete (const char *name, StringList &matches)
{
    uint32_t i;
    if (name && name[0])
    {
        for (i = 0; i < ArchSpec::kNumCores; ++i)
        {
            if (NameMatches(g_core_definitions[i].name, eNameMatchStartsWith, name))
                matches.AppendString (g_core_definitions[i].name);
        }
    }
    else
    {
        for (i = 0; i < ArchSpec::kNumCores; ++i)
            matches.AppendString (g_core_definitions[i].name);
    }
    return matches.GetSize();
}



#define CPU_ANY (UINT32_MAX)

//===----------------------------------------------------------------------===//
// A table that gets searched linearly for matches. This table is used to
// convert cpu type and subtypes to architecture names, and to convert
// architecture names to cpu types and subtypes. The ordering is important and
// allows the precedence to be set when the table is built.
#define SUBTYPE_MASK 0x00FFFFFFu
static const ArchDefinitionEntry g_macho_arch_entries[] =
{
    { ArchSpec::eCore_arm_generic     , llvm::MachO::CPUTypeARM       , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    { ArchSpec::eCore_arm_generic     , llvm::MachO::CPUTypeARM       , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv4       , llvm::MachO::CPUTypeARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv4t      , llvm::MachO::CPUTypeARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv6       , llvm::MachO::CPUTypeARM       , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5       , llvm::MachO::CPUTypeARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5e      , llvm::MachO::CPUTypeARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5t      , llvm::MachO::CPUTypeARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_xscale      , llvm::MachO::CPUTypeARM       , 8      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7       , llvm::MachO::CPUTypeARM       , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7f      , llvm::MachO::CPUTypeARM       , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7k      , llvm::MachO::CPUTypeARM       , 12     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7s      , llvm::MachO::CPUTypeARM       , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumb           , llvm::MachO::CPUTypeARM       , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv4t        , llvm::MachO::CPUTypeARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv5         , llvm::MachO::CPUTypeARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv5e        , llvm::MachO::CPUTypeARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv6         , llvm::MachO::CPUTypeARM       , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7         , llvm::MachO::CPUTypeARM       , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7f        , llvm::MachO::CPUTypeARM       , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7k        , llvm::MachO::CPUTypeARM       , 12     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7s        , llvm::MachO::CPUTypeARM       , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_generic     , llvm::MachO::CPUTypePowerPC   , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    { ArchSpec::eCore_ppc_generic     , llvm::MachO::CPUTypePowerPC   , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc601      , llvm::MachO::CPUTypePowerPC   , 1      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc602      , llvm::MachO::CPUTypePowerPC   , 2      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603      , llvm::MachO::CPUTypePowerPC   , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603e     , llvm::MachO::CPUTypePowerPC   , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603ev    , llvm::MachO::CPUTypePowerPC   , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc604      , llvm::MachO::CPUTypePowerPC   , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc604e     , llvm::MachO::CPUTypePowerPC   , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc620      , llvm::MachO::CPUTypePowerPC   , 8      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc750      , llvm::MachO::CPUTypePowerPC   , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc7400     , llvm::MachO::CPUTypePowerPC   , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc7450     , llvm::MachO::CPUTypePowerPC   , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc970      , llvm::MachO::CPUTypePowerPC   , 100    , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc64_generic   , llvm::MachO::CPUTypePowerPC64 , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc64_ppc970_64 , llvm::MachO::CPUTypePowerPC64 , 100    , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i386     , llvm::MachO::CPUTypeI386      , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i486     , llvm::MachO::CPUTypeI386      , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i486sx   , llvm::MachO::CPUTypeI386      , 0x84   , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i386     , llvm::MachO::CPUTypeI386      , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPUTypeX86_64    , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPUTypeX86_64    , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPUTypeX86_64    , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    // Catch any unknown mach architectures so we can always use the object and symbol mach-o files
    { ArchSpec::eCore_uknownMach32    , 0                             , 0      , 0xFF000000u, 0x00000000u },
    { ArchSpec::eCore_uknownMach64    , llvm::MachO::CPUArchABI64     , 0      , 0xFF000000u, 0x00000000u }
};
static const ArchDefinition g_macho_arch_def = {
    eArchTypeMachO,
    sizeof(g_macho_arch_entries)/sizeof(g_macho_arch_entries[0]),
    g_macho_arch_entries,
    "mach-o"
};

//===----------------------------------------------------------------------===//
// A table that gets searched linearly for matches. This table is used to
// convert cpu type and subtypes to architecture names, and to convert
// architecture names to cpu types and subtypes. The ordering is important and
// allows the precedence to be set when the table is built.
static const ArchDefinitionEntry g_elf_arch_entries[] =
{
    { ArchSpec::eCore_sparc_generic   , llvm::ELF::EM_SPARC  , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // Sparc
    { ArchSpec::eCore_x86_32_i386     , llvm::ELF::EM_386    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // Intel 80386
    { ArchSpec::eCore_x86_32_i486     , llvm::ELF::EM_486    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // Intel 486 (deprecated)
    { ArchSpec::eCore_ppc_generic     , llvm::ELF::EM_PPC    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC
    { ArchSpec::eCore_ppc64_generic   , llvm::ELF::EM_PPC64  , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC64
    { ArchSpec::eCore_arm_generic     , llvm::ELF::EM_ARM    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARM
    { ArchSpec::eCore_sparc9_generic  , llvm::ELF::EM_SPARCV9, LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // SPARC V9
    { ArchSpec::eCore_x86_64_x86_64   , llvm::ELF::EM_X86_64 , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }  // AMD64
};

static const ArchDefinition g_elf_arch_def = {
    eArchTypeELF,
    sizeof(g_elf_arch_entries)/sizeof(g_elf_arch_entries[0]),
    g_elf_arch_entries,
    "elf",
};

//===----------------------------------------------------------------------===//
// Table of all ArchDefinitions
static const ArchDefinition *g_arch_definitions[] = {
    &g_macho_arch_def,
    &g_elf_arch_def
};

static const size_t k_num_arch_definitions =
    sizeof(g_arch_definitions) / sizeof(g_arch_definitions[0]);

//===----------------------------------------------------------------------===//
// Static helper functions.


// Get the architecture definition for a given object type.
static const ArchDefinition *
FindArchDefinition (ArchitectureType arch_type)
{
    for (unsigned int i = 0; i < k_num_arch_definitions; ++i)
    {
        const ArchDefinition *def = g_arch_definitions[i];
        if (def->type == arch_type)
            return def;
    }
    return NULL;
}

// Get an architecture definition by name.
static const CoreDefinition *
FindCoreDefinition (llvm::StringRef name)
{
    for (unsigned int i = 0; i < ArchSpec::kNumCores; ++i)
    {
        if (name.equals_lower(g_core_definitions[i].name))
            return &g_core_definitions[i];
    }
    return NULL;
}

static inline const CoreDefinition *
FindCoreDefinition (ArchSpec::Core core)
{
    if (core >= 0 && core < ArchSpec::kNumCores)
        return &g_core_definitions[core];
    return NULL;
}

// Get a definition entry by cpu type and subtype.
static const ArchDefinitionEntry *
FindArchDefinitionEntry (const ArchDefinition *def, uint32_t cpu, uint32_t sub)
{
    if (def == NULL)
        return NULL;

    const ArchDefinitionEntry *entries = def->entries;
    for (size_t i = 0; i < def->num_entries; ++i)
    {
        if (entries[i].cpu == (cpu & entries[i].cpu_mask))
            if (entries[i].sub == (sub & entries[i].sub_mask))
                return &entries[i];
    }
    return NULL;
}

static const ArchDefinitionEntry *
FindArchDefinitionEntry (const ArchDefinition *def, ArchSpec::Core core)
{
    if (def == NULL)
        return NULL;
    
    const ArchDefinitionEntry *entries = def->entries;
    for (size_t i = 0; i < def->num_entries; ++i)
    {
        if (entries[i].core == core)
            return &entries[i];
    }
    return NULL;
}

//===----------------------------------------------------------------------===//
// Constructors and destructors.

ArchSpec::ArchSpec() :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid)
{
}

ArchSpec::ArchSpec (const char *triple_cstr, Platform *platform) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid)
{
    if (triple_cstr)
        SetTriple(triple_cstr, platform);
}


ArchSpec::ArchSpec (const char *triple_cstr) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid)
{
    if (triple_cstr)
        SetTriple(triple_cstr);
}

ArchSpec::ArchSpec(const llvm::Triple &triple) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid)
{
    SetTriple(triple);
}

ArchSpec::ArchSpec (ArchitectureType arch_type, uint32_t cpu, uint32_t subtype) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid)
{
    SetArchitecture (arch_type, cpu, subtype);
}

ArchSpec::~ArchSpec()
{
}

//===----------------------------------------------------------------------===//
// Assignment and initialization.

const ArchSpec&
ArchSpec::operator= (const ArchSpec& rhs)
{
    if (this != &rhs)
    {
        m_triple = rhs.m_triple;
        m_core = rhs.m_core;
        m_byte_order = rhs.m_byte_order;
    }
    return *this;
}

void
ArchSpec::Clear()
{
    m_triple = llvm::Triple();
    m_core = kCore_invalid;
    m_byte_order = eByteOrderInvalid;
}

//===----------------------------------------------------------------------===//
// Predicates.


const char *
ArchSpec::GetArchitectureName () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->name;
    return "unknown";
}

uint32_t
ArchSpec::GetMachOCPUType () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
    {
        const ArchDefinitionEntry *arch_def = FindArchDefinitionEntry (&g_macho_arch_def, core_def->core);
        if (arch_def)
        {
            return arch_def->cpu;
        }
    }
    return LLDB_INVALID_CPUTYPE;
}

uint32_t
ArchSpec::GetMachOCPUSubType () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
    {
        const ArchDefinitionEntry *arch_def = FindArchDefinitionEntry (&g_macho_arch_def, core_def->core);
        if (arch_def)
        {
            return arch_def->sub;
        }
    }
    return LLDB_INVALID_CPUTYPE;
}

llvm::Triple::ArchType
ArchSpec::GetMachine () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->machine;

    return llvm::Triple::UnknownArch;
}

uint32_t
ArchSpec::GetAddressByteSize() const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->addr_byte_size;
    return 0;
}

ByteOrder
ArchSpec::GetDefaultEndian () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->default_byte_order;
    return eByteOrderInvalid;
}

lldb::ByteOrder
ArchSpec::GetByteOrder () const
{
    if (m_byte_order == eByteOrderInvalid)
        return GetDefaultEndian();
    return m_byte_order;
}

//===----------------------------------------------------------------------===//
// Mutators.

bool
ArchSpec::SetTriple (const llvm::Triple &triple)
{
    m_triple = triple;
    
    llvm::StringRef arch_name (m_triple.getArchName());
    const CoreDefinition *core_def = FindCoreDefinition (arch_name);
    if (core_def)
    {
        m_core = core_def->core;
        // Set the byte order to the default byte order for an architecture.
        // This can be modified if needed for cases when cores handle both
        // big and little endian
        m_byte_order = core_def->default_byte_order; 
    }
    else
    {
        Clear();
    }

    
    return IsValid();
}

static bool
ParseMachCPUDashSubtypeTriple (const char *triple_cstr, ArchSpec &arch)
{
    // Accept "12-10" or "12.10" as cpu type/subtype
    if (isdigit(triple_cstr[0]))
    {
        char *end = NULL;
        errno = 0;
        uint32_t cpu = ::strtoul (triple_cstr, &end, 0);
        if (errno == 0 && cpu != 0 && end && ((*end == '-') || (*end == '.')))
        {
            errno = 0;
            uint32_t sub = ::strtoul (end + 1, &end, 0);
            if (errno == 0 && end && ((*end == '-') || (*end == '.') || (*end == '\0')))
            {
                if (arch.SetArchitecture (eArchTypeMachO, cpu, sub))
                {
                    if (*end == '-')
                    {
                        llvm::StringRef vendor_os (end + 1);
                        size_t dash_pos = vendor_os.find('-');
                        if (dash_pos != llvm::StringRef::npos)
                        {
                            llvm::StringRef vendor_str(vendor_os.substr(0, dash_pos));
                            arch.GetTriple().setVendorName(vendor_str);
                            const size_t vendor_start_pos = dash_pos+1;
                            dash_pos = vendor_os.find(vendor_start_pos, '-');
                            if (dash_pos == llvm::StringRef::npos)
                            {
                                if (vendor_start_pos < vendor_os.size())
                                    arch.GetTriple().setOSName(vendor_os.substr(vendor_start_pos));
                            }
                            else
                            {
                                arch.GetTriple().setOSName(vendor_os.substr(vendor_start_pos, dash_pos - vendor_start_pos));
                            }
                        }
                    }
                    return true;
                }
            }
        }
    }
    return false;
}
bool
ArchSpec::SetTriple (const char *triple_cstr)
{
    if (triple_cstr && triple_cstr[0])
    {
        if (ParseMachCPUDashSubtypeTriple (triple_cstr, *this))
            return true;
        
        llvm::StringRef triple_stref (triple_cstr);
        if (triple_stref.startswith (LLDB_ARCH_DEFAULT))
        {
            // Special case for the current host default architectures...
            if (triple_stref.equals (LLDB_ARCH_DEFAULT_32BIT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT_64BIT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture64);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        }
        else
        {
            std::string normalized_triple_sstr (llvm::Triple::normalize(triple_stref));
            triple_stref = normalized_triple_sstr;
            SetTriple (llvm::Triple (triple_stref));
        }
    }
    else
        Clear();
    return IsValid();
}

bool
ArchSpec::SetTriple (const char *triple_cstr, Platform *platform)
{
    if (triple_cstr && triple_cstr[0])
    {
        if (ParseMachCPUDashSubtypeTriple (triple_cstr, *this))
            return true;
        
        llvm::StringRef triple_stref (triple_cstr);
        if (triple_stref.startswith (LLDB_ARCH_DEFAULT))
        {
            // Special case for the current host default architectures...
            if (triple_stref.equals (LLDB_ARCH_DEFAULT_32BIT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT_64BIT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture64);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT))
                *this = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        }
        else
        {
            ArchSpec raw_arch (triple_cstr);

            std::string normalized_triple_sstr (llvm::Triple::normalize(triple_stref));
            triple_stref = normalized_triple_sstr;
            llvm::Triple normalized_triple (triple_stref);
            
            const bool os_specified = normalized_triple.getOSName().size() > 0;
            const bool vendor_specified = normalized_triple.getVendorName().size() > 0;
            const bool env_specified = normalized_triple.getEnvironmentName().size() > 0;
            
            // If we got an arch only, then default the vendor, os, environment 
            // to match the platform if one is supplied
            if (!(os_specified || vendor_specified || env_specified))
            {
                if (platform)
                {
                    // If we were given a platform, use the platform's system
                    // architecture. If this is not available (might not be
                    // connected) use the first supported architecture.
                    ArchSpec compatible_arch;
                    if (platform->IsCompatibleArchitecture (raw_arch, &compatible_arch))
                    {
                        if (compatible_arch.IsValid())
                        {
                            const llvm::Triple &compatible_triple = compatible_arch.GetTriple();
                            if (!vendor_specified)
                                normalized_triple.setVendor(compatible_triple.getVendor());
                            if (!os_specified)
                                normalized_triple.setOS(compatible_triple.getOS());
                            if (!env_specified && compatible_triple.getEnvironmentName().size())
                                normalized_triple.setEnvironment(compatible_triple.getEnvironment());
                        }
                    }
                    else
                    {
                        *this = raw_arch;
                        return IsValid();
                    }
                }
                else
                {
                    // No platform specified, fall back to the host system for
                    // the default vendor, os, and environment.
                    llvm::Triple host_triple(llvm::sys::getDefaultTargetTriple());
                    if (!vendor_specified)
                        normalized_triple.setVendor(host_triple.getVendor());
                    if (!vendor_specified)
                        normalized_triple.setOS(host_triple.getOS());
                    if (!env_specified && host_triple.getEnvironmentName().size())
                        normalized_triple.setEnvironment(host_triple.getEnvironment());
                }
            }
            SetTriple (normalized_triple);
        }
    }
    else
        Clear();
    return IsValid();
}

bool
ArchSpec::SetArchitecture (ArchitectureType arch_type, uint32_t cpu, uint32_t sub)
{
    m_core = kCore_invalid;
    bool update_triple = true;
    const ArchDefinition *arch_def = FindArchDefinition(arch_type);
    if (arch_def)
    {
        const ArchDefinitionEntry *arch_def_entry = FindArchDefinitionEntry (arch_def, cpu, sub);
        if (arch_def_entry)
        {
            const CoreDefinition *core_def = FindCoreDefinition (arch_def_entry->core);
            if (core_def)
            {
                m_core = core_def->core;
                update_triple = false;
                // Always use the architecture name because it might be more descriptive
                // than the architecture enum ("armv7" -> llvm::Triple::arm).
                m_triple.setArchName(llvm::StringRef(core_def->name));
                if (arch_type == eArchTypeMachO)
                {
                    m_triple.setVendor (llvm::Triple::Apple);

                    switch (core_def->machine)
                    {
                        case llvm::Triple::arm:
                        case llvm::Triple::thumb:
                            m_triple.setOS (llvm::Triple::IOS);
                            break;
                            
                        case llvm::Triple::x86:
                        case llvm::Triple::x86_64:
                        default:
                            m_triple.setOS (llvm::Triple::MacOSX);
                            break;
                    }
                }
                else
                {
                    m_triple.setVendor (llvm::Triple::UnknownVendor);
                    m_triple.setOS (llvm::Triple::UnknownOS);
                }
                // Fall back onto setting the machine type if the arch by name failed...
                if (m_triple.getArch () == llvm::Triple::UnknownArch)
                    m_triple.setArch (core_def->machine);
            }
        }
    }
    CoreUpdated(update_triple);
    return IsValid();
}

uint32_t
ArchSpec::GetMinimumOpcodeByteSize() const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->min_opcode_byte_size;
    return 0;
}

uint32_t
ArchSpec::GetMaximumOpcodeByteSize() const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->max_opcode_byte_size;
    return 0;
}

bool
ArchSpec::IsExactMatch (const ArchSpec& rhs) const
{
    return Compare (rhs, true);
}

bool
ArchSpec::IsCompatibleMatch (const ArchSpec& rhs) const
{
    return Compare (rhs, false);
}

bool
ArchSpec::Compare (const ArchSpec& rhs, bool exact_match) const
{
    if (GetByteOrder() != rhs.GetByteOrder())
        return false;
        
    const ArchSpec::Core lhs_core = GetCore ();
    const ArchSpec::Core rhs_core = rhs.GetCore ();

    const bool core_match = cores_match (lhs_core, rhs_core, true, exact_match);

    if (core_match)
    {
        const llvm::Triple &lhs_triple = GetTriple();
        const llvm::Triple &rhs_triple = rhs.GetTriple();

        const llvm::Triple::VendorType lhs_triple_vendor = lhs_triple.getVendor();
        const llvm::Triple::VendorType rhs_triple_vendor = rhs_triple.getVendor();
        if (lhs_triple_vendor != rhs_triple_vendor)
        {
            const bool rhs_vendor_specified = rhs.TripleVendorWasSpecified();
            const bool lhs_vendor_specified = TripleVendorWasSpecified();
            // Both architectures had the vendor specified, so if they aren't
            // equal then we return false
            if (rhs_vendor_specified && lhs_vendor_specified)
                return false;
            
            // Only fail if both vendor types are not unknown
            if (lhs_triple_vendor != llvm::Triple::UnknownVendor &&
                rhs_triple_vendor != llvm::Triple::UnknownVendor)
                return false;
        }
        
        const llvm::Triple::OSType lhs_triple_os = lhs_triple.getOS();
        const llvm::Triple::OSType rhs_triple_os = rhs_triple.getOS();
        if (lhs_triple_os != rhs_triple_os)
        {
            const bool rhs_os_specified = rhs.TripleOSWasSpecified();
            const bool lhs_os_specified = TripleOSWasSpecified();
            // Both architectures had the OS specified, so if they aren't
            // equal then we return false
            if (rhs_os_specified && lhs_os_specified)
                return false;
            // Only fail if both os types are not unknown
            if (lhs_triple_os != llvm::Triple::UnknownOS &&
                rhs_triple_os != llvm::Triple::UnknownOS)
                return false;
        }

        const llvm::Triple::EnvironmentType lhs_triple_env = lhs_triple.getEnvironment();
        const llvm::Triple::EnvironmentType rhs_triple_env = rhs_triple.getEnvironment();
            
        if (lhs_triple_env != rhs_triple_env)
        {
            // Only fail if both environment types are not unknown
            if (lhs_triple_env != llvm::Triple::UnknownEnvironment &&
                rhs_triple_env != llvm::Triple::UnknownEnvironment)
                return false;
        }
        return true;
    }
    return false;
}

//===----------------------------------------------------------------------===//
// Helper methods.

void
ArchSpec::CoreUpdated (bool update_triple)
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
    {
        if (update_triple)
            m_triple = llvm::Triple(core_def->name, "unknown", "unknown");
        m_byte_order = core_def->default_byte_order;
    }
    else
    {
        if (update_triple)
            m_triple = llvm::Triple();
        m_byte_order = eByteOrderInvalid;
    }
}

//===----------------------------------------------------------------------===//
// Operators.

static bool
cores_match (const ArchSpec::Core core1, const ArchSpec::Core core2, bool try_inverse, bool enforce_exact_match)
{
    if (core1 == core2)
        return true;

    switch (core1)
    {
    case ArchSpec::kCore_any:
        return true;

    case ArchSpec::kCore_arm_any:
        if (core2 >= ArchSpec::kCore_arm_first && core2 <= ArchSpec::kCore_arm_last)
            return true;
        if (core2 >= ArchSpec::kCore_thumb_first && core2 <= ArchSpec::kCore_thumb_last)
            return true;
        if (core2 == ArchSpec::kCore_arm_any)
            return true;
        break;
        
    case ArchSpec::kCore_x86_32_any:
        if ((core2 >= ArchSpec::kCore_x86_32_first && core2 <= ArchSpec::kCore_x86_32_last) || (core2 == ArchSpec::kCore_x86_32_any))
            return true;
        break;
        
    case ArchSpec::kCore_ppc_any:
        if ((core2 >= ArchSpec::kCore_ppc_first && core2 <= ArchSpec::kCore_ppc_last) || (core2 == ArchSpec::kCore_ppc_any))
            return true;
        break;
        
    case ArchSpec::kCore_ppc64_any:
        if ((core2 >= ArchSpec::kCore_ppc64_first && core2 <= ArchSpec::kCore_ppc64_last) || (core2 == ArchSpec::kCore_ppc64_any))
            return true;
        break;

    case ArchSpec::eCore_arm_armv7f:
    case ArchSpec::eCore_arm_armv7k:
    case ArchSpec::eCore_arm_armv7s:
        if (!enforce_exact_match)
        {
            try_inverse = false;
            if (core2 == ArchSpec::eCore_arm_armv7)
                return true;
        }
        break;

    default:
        break;
    }
    if (try_inverse)
        return cores_match (core2, core1, false, enforce_exact_match);
    return false;
}

bool
lldb_private::operator== (const ArchSpec& lhs, const ArchSpec& rhs)
{
    return lhs.IsExactMatch (rhs);
}

bool
lldb_private::operator!= (const ArchSpec& lhs, const ArchSpec& rhs)
{
    return !(lhs == rhs);
}

bool
lldb_private::operator<(const ArchSpec& lhs, const ArchSpec& rhs)
{
    const ArchSpec::Core lhs_core = lhs.GetCore ();
    const ArchSpec::Core rhs_core = rhs.GetCore ();
    return lhs_core < rhs_core;
}
