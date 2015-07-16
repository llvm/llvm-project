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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Host.h"

#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StringList.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/NameMatches.h"
#include "lldb/Utility/SafeMachO.h"
#include "Plugins/Process/Utility/ARMDefines.h"
#include "Plugins/Process/Utility/InstructionUtils.h"

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
        const char * const name;
    };

}

// This core information can be looked using the ArchSpec::Core as the index
static const CoreDefinition g_core_definitions[] =
{
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_generic     , "arm"       },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv4       , "armv4"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv4t      , "armv4t"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5       , "armv5"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5e      , "armv5e"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv5t      , "armv5t"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv6       , "armv6"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv6m      , "armv6m"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7       , "armv7"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7f      , "armv7f"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7s      , "armv7s"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7k      , "armv7k"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7m      , "armv7m"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_armv7em     , "armv7em"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::arm    , ArchSpec::eCore_arm_xscale      , "xscale"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumb           , "thumb"     },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv4t        , "thumbv4t"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv5         , "thumbv5"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv5e        , "thumbv5e"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv6         , "thumbv6"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv6m        , "thumbv6m"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7         , "thumbv7"   },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7f        , "thumbv7f"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7s        , "thumbv7s"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7k        , "thumbv7k"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7m        , "thumbv7m"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::thumb  , ArchSpec::eCore_thumbv7em       , "thumbv7em" },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64, ArchSpec::eCore_arm_arm64       , "arm64"     },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64, ArchSpec::eCore_arm_armv8       , "armv8"     },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::aarch64, ArchSpec::eCore_arm_aarch64     , "aarch64"   },

    // mips32, mips32r2, mips32r3, mips32r5, mips32r6
    { eByteOrderBig   , 4, 2, 4, llvm::Triple::mips  , ArchSpec::eCore_mips32         , "mips"      },
    { eByteOrderBig   , 4, 2, 4, llvm::Triple::mips  , ArchSpec::eCore_mips32r2       , "mipsr2"    },
    { eByteOrderBig   , 4, 2, 4, llvm::Triple::mips  , ArchSpec::eCore_mips32r3       , "mipsr3"    },
    { eByteOrderBig   , 4, 2, 4, llvm::Triple::mips  , ArchSpec::eCore_mips32r5       , "mipsr5"    },
    { eByteOrderBig   , 4, 2, 4, llvm::Triple::mips  , ArchSpec::eCore_mips32r6       , "mipsr6"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32el       , "mipsel"    },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32r2el     , "mipsr2el"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32r3el     , "mipsr3el"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32r5el     , "mipsr5el"  },
    { eByteOrderLittle, 4, 2, 4, llvm::Triple::mipsel, ArchSpec::eCore_mips32r6el     , "mipsr6el"  },
    
    // mips64, mips64r2, mips64r3, mips64r5, mips64r6
    { eByteOrderBig   , 8, 2, 4, llvm::Triple::mips64  , ArchSpec::eCore_mips64         , "mips64"      },
    { eByteOrderBig   , 8, 2, 4, llvm::Triple::mips64  , ArchSpec::eCore_mips64r2       , "mips64r2"    },
    { eByteOrderBig   , 8, 2, 4, llvm::Triple::mips64  , ArchSpec::eCore_mips64r3       , "mips64r3"    },
    { eByteOrderBig   , 8, 2, 4, llvm::Triple::mips64  , ArchSpec::eCore_mips64r5       , "mips64r5"    },
    { eByteOrderBig   , 8, 2, 4, llvm::Triple::mips64  , ArchSpec::eCore_mips64r6       , "mips64r6"    },
    { eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el, ArchSpec::eCore_mips64el       , "mips64el"    },
    { eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el, ArchSpec::eCore_mips64r2el     , "mips64r2el"  },
    { eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el, ArchSpec::eCore_mips64r3el     , "mips64r3el"  },
    { eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el, ArchSpec::eCore_mips64r5el     , "mips64r5el"  },
    { eByteOrderLittle, 8, 2, 4, llvm::Triple::mips64el, ArchSpec::eCore_mips64r6el     , "mips64r6el"  },
    
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_generic     , "powerpc"   },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc601      , "ppc601"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc602      , "ppc602"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603      , "ppc603"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603e     , "ppc603e"   },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc603ev    , "ppc603ev"  },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc604      , "ppc604"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc604e     , "ppc604e"   },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc620      , "ppc620"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc750      , "ppc750"    },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc7400     , "ppc7400"   },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc7450     , "ppc7450"   },
    { eByteOrderBig   , 4, 4, 4, llvm::Triple::ppc    , ArchSpec::eCore_ppc_ppc970      , "ppc970"    },
    
    { eByteOrderBig   , 8, 4, 4, llvm::Triple::ppc64  , ArchSpec::eCore_ppc64_generic   , "powerpc64" },
    { eByteOrderBig   , 8, 4, 4, llvm::Triple::ppc64  , ArchSpec::eCore_ppc64_ppc970_64 , "ppc970-64" },
    
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::sparc  , ArchSpec::eCore_sparc_generic   , "sparc"     },
    { eByteOrderLittle, 8, 4, 4, llvm::Triple::sparcv9, ArchSpec::eCore_sparc9_generic  , "sparcv9"   },

    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i386    , "i386"      },
    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i486    , "i486"      },
    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i486sx  , "i486sx"    },
    { eByteOrderLittle, 4, 1, 15, llvm::Triple::x86    , ArchSpec::eCore_x86_32_i686    , "i686"      },

    { eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64 , ArchSpec::eCore_x86_64_x86_64  , "x86_64"    },
    { eByteOrderLittle, 8, 1, 15, llvm::Triple::x86_64 , ArchSpec::eCore_x86_64_x86_64h , "x86_64h"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon , ArchSpec::eCore_hexagon_generic,    "hexagon"   },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon , ArchSpec::eCore_hexagon_hexagonv4,  "hexagonv4" },
    { eByteOrderLittle, 4, 4, 4, llvm::Triple::hexagon , ArchSpec::eCore_hexagon_hexagonv5,  "hexagonv5" },

    { eByteOrderLittle, 4, 4, 4 , llvm::Triple::UnknownArch , ArchSpec::eCore_uknownMach32  , "unknown-mach-32" },
    { eByteOrderLittle, 8, 4, 4 , llvm::Triple::UnknownArch , ArchSpec::eCore_uknownMach64  , "unknown-mach-64" },

    { eByteOrderBig   , 4, 1, 1 , llvm::Triple::kalimba , ArchSpec::eCore_kalimba3  , "kalimba3" },
    { eByteOrderLittle, 4, 1, 1 , llvm::Triple::kalimba , ArchSpec::eCore_kalimba4  , "kalimba4" },
    { eByteOrderLittle, 4, 1, 1 , llvm::Triple::kalimba , ArchSpec::eCore_kalimba5  , "kalimba5" }
};

// Ensure that we have an entry in the g_core_definitions for each core. If you comment out an entry above,
// you will need to comment out the corresponding ArchSpec::Core enumeration.
static_assert(sizeof(g_core_definitions) / sizeof(CoreDefinition) == ArchSpec::kNumCores, "make sure we have one core definition for each core");


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


size_t
ArchSpec::AutoComplete (const char *name, StringList &matches)
{
    uint32_t i;
    if (name && name[0])
    {
        for (i = 0; i < llvm::array_lengthof(g_core_definitions); ++i)
        {
            if (NameMatches(g_core_definitions[i].name, eNameMatchStartsWith, name))
                matches.AppendString (g_core_definitions[i].name);
        }
    }
    else
    {
        for (i = 0; i < llvm::array_lengthof(g_core_definitions); ++i)
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
    { ArchSpec::eCore_arm_generic     , llvm::MachO::CPU_TYPE_ARM       , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    { ArchSpec::eCore_arm_generic     , llvm::MachO::CPU_TYPE_ARM       , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv4       , llvm::MachO::CPU_TYPE_ARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv4t      , llvm::MachO::CPU_TYPE_ARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv6       , llvm::MachO::CPU_TYPE_ARM       , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv6m      , llvm::MachO::CPU_TYPE_ARM       , 14     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5       , llvm::MachO::CPU_TYPE_ARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5e      , llvm::MachO::CPU_TYPE_ARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv5t      , llvm::MachO::CPU_TYPE_ARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_xscale      , llvm::MachO::CPU_TYPE_ARM       , 8      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7       , llvm::MachO::CPU_TYPE_ARM       , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7f      , llvm::MachO::CPU_TYPE_ARM       , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7s      , llvm::MachO::CPU_TYPE_ARM       , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7k      , llvm::MachO::CPU_TYPE_ARM       , 12     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7m      , llvm::MachO::CPU_TYPE_ARM       , 15     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_armv7em     , llvm::MachO::CPU_TYPE_ARM       , 16     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_arm64       , llvm::MachO::CPU_TYPE_ARM64     , 1      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_arm64       , llvm::MachO::CPU_TYPE_ARM64     , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_arm64       , llvm::MachO::CPU_TYPE_ARM64     , 13     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_arm_arm64       , llvm::MachO::CPU_TYPE_ARM64     , CPU_ANY, UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumb           , llvm::MachO::CPU_TYPE_ARM       , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv4t        , llvm::MachO::CPU_TYPE_ARM       , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv5         , llvm::MachO::CPU_TYPE_ARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv5e        , llvm::MachO::CPU_TYPE_ARM       , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv6         , llvm::MachO::CPU_TYPE_ARM       , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv6m        , llvm::MachO::CPU_TYPE_ARM       , 14     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7         , llvm::MachO::CPU_TYPE_ARM       , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7f        , llvm::MachO::CPU_TYPE_ARM       , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7s        , llvm::MachO::CPU_TYPE_ARM       , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7k        , llvm::MachO::CPU_TYPE_ARM       , 12     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7m        , llvm::MachO::CPU_TYPE_ARM       , 15     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_thumbv7em       , llvm::MachO::CPU_TYPE_ARM       , 16     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_generic     , llvm::MachO::CPU_TYPE_POWERPC   , CPU_ANY, UINT32_MAX , UINT32_MAX  },
    { ArchSpec::eCore_ppc_generic     , llvm::MachO::CPU_TYPE_POWERPC   , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc601      , llvm::MachO::CPU_TYPE_POWERPC   , 1      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc602      , llvm::MachO::CPU_TYPE_POWERPC   , 2      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603      , llvm::MachO::CPU_TYPE_POWERPC   , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603e     , llvm::MachO::CPU_TYPE_POWERPC   , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc603ev    , llvm::MachO::CPU_TYPE_POWERPC   , 5      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc604      , llvm::MachO::CPU_TYPE_POWERPC   , 6      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc604e     , llvm::MachO::CPU_TYPE_POWERPC   , 7      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc620      , llvm::MachO::CPU_TYPE_POWERPC   , 8      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc750      , llvm::MachO::CPU_TYPE_POWERPC   , 9      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc7400     , llvm::MachO::CPU_TYPE_POWERPC   , 10     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc7450     , llvm::MachO::CPU_TYPE_POWERPC   , 11     , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc_ppc970      , llvm::MachO::CPU_TYPE_POWERPC   , 100    , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc64_generic   , llvm::MachO::CPU_TYPE_POWERPC64 , 0      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_ppc64_ppc970_64 , llvm::MachO::CPU_TYPE_POWERPC64 , 100    , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i386     , llvm::MachO::CPU_TYPE_I386      , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i486     , llvm::MachO::CPU_TYPE_I386      , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i486sx   , llvm::MachO::CPU_TYPE_I386      , 0x84   , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_32_i386     , llvm::MachO::CPU_TYPE_I386      , CPU_ANY, UINT32_MAX , UINT32_MAX   },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPU_TYPE_X86_64    , 3      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPU_TYPE_X86_64    , 4      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_64_x86_64h  , llvm::MachO::CPU_TYPE_X86_64    , 8      , UINT32_MAX , SUBTYPE_MASK },
    { ArchSpec::eCore_x86_64_x86_64   , llvm::MachO::CPU_TYPE_X86_64    , CPU_ANY, UINT32_MAX , UINT32_MAX   },
    // Catch any unknown mach architectures so we can always use the object and symbol mach-o files
    { ArchSpec::eCore_uknownMach32    , 0                               , 0      , 0xFF000000u, 0x00000000u },
    { ArchSpec::eCore_uknownMach64    , llvm::MachO::CPU_ARCH_ABI64     , 0      , 0xFF000000u, 0x00000000u }
};
static const ArchDefinition g_macho_arch_def = {
    eArchTypeMachO,
    llvm::array_lengthof(g_macho_arch_entries),
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
    { ArchSpec::eCore_x86_32_i486     , llvm::ELF::EM_IAMCU  , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // Intel MCU // FIXME: is this correct?
    { ArchSpec::eCore_ppc_generic     , llvm::ELF::EM_PPC    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC
    { ArchSpec::eCore_ppc64_generic   , llvm::ELF::EM_PPC64  , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC64
    { ArchSpec::eCore_arm_generic     , llvm::ELF::EM_ARM    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARM
    { ArchSpec::eCore_arm_aarch64     , llvm::ELF::EM_AARCH64, LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARM64
    { ArchSpec::eCore_sparc9_generic  , llvm::ELF::EM_SPARCV9, LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // SPARC V9
    { ArchSpec::eCore_x86_64_x86_64   , llvm::ELF::EM_X86_64 , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // AMD64
    { ArchSpec::eCore_mips32          , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32,     0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32
    { ArchSpec::eCore_mips32r2        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32r2,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32r2
    { ArchSpec::eCore_mips32r6        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32r6,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32r6
    { ArchSpec::eCore_mips32el        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32el,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32el
    { ArchSpec::eCore_mips32r2el      , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32r2el, 0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32r2el
    { ArchSpec::eCore_mips32r6el      , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips32r6el, 0xFFFFFFFFu, 0xFFFFFFFFu }, // mips32r6el
    { ArchSpec::eCore_mips64          , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64,     0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64
    { ArchSpec::eCore_mips64r2        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64r2,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64r2
    { ArchSpec::eCore_mips64r6        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64r6,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64r6
    { ArchSpec::eCore_mips64el        , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64el,   0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64el
    { ArchSpec::eCore_mips64r2el      , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64r2el, 0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64r2el
    { ArchSpec::eCore_mips64r6el      , llvm::ELF::EM_MIPS   , ArchSpec::eMIPSSubType_mips64r6el, 0xFFFFFFFFu, 0xFFFFFFFFu }, // mips64r6el
    { ArchSpec::eCore_hexagon_generic , llvm::ELF::EM_HEXAGON, LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // HEXAGON
    { ArchSpec::eCore_kalimba3 ,        llvm::ELF::EM_CSR_KALIMBA, llvm::Triple::KalimbaSubArch_v3, 0xFFFFFFFFu, 0xFFFFFFFFu },  // KALIMBA
    { ArchSpec::eCore_kalimba4 ,        llvm::ELF::EM_CSR_KALIMBA, llvm::Triple::KalimbaSubArch_v4, 0xFFFFFFFFu, 0xFFFFFFFFu },  // KALIMBA
    { ArchSpec::eCore_kalimba5 ,        llvm::ELF::EM_CSR_KALIMBA, llvm::Triple::KalimbaSubArch_v5, 0xFFFFFFFFu, 0xFFFFFFFFu }  // KALIMBA
};

static const ArchDefinition g_elf_arch_def = {
    eArchTypeELF,
    llvm::array_lengthof(g_elf_arch_entries),
    g_elf_arch_entries,
    "elf",
};

static const ArchDefinitionEntry g_coff_arch_entries[] =
{
    { ArchSpec::eCore_x86_32_i386  , llvm::COFF::IMAGE_FILE_MACHINE_I386     , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // Intel 80x86
    { ArchSpec::eCore_ppc_generic  , llvm::COFF::IMAGE_FILE_MACHINE_POWERPC  , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC
    { ArchSpec::eCore_ppc_generic  , llvm::COFF::IMAGE_FILE_MACHINE_POWERPCFP, LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // PowerPC (with FPU)
    { ArchSpec::eCore_arm_generic  , llvm::COFF::IMAGE_FILE_MACHINE_ARM      , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARM
    { ArchSpec::eCore_arm_armv7    , llvm::COFF::IMAGE_FILE_MACHINE_ARMNT    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARMv7
    { ArchSpec::eCore_thumb        , llvm::COFF::IMAGE_FILE_MACHINE_THUMB    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }, // ARMv7
    { ArchSpec::eCore_x86_64_x86_64, llvm::COFF::IMAGE_FILE_MACHINE_AMD64    , LLDB_INVALID_CPUTYPE, 0xFFFFFFFFu, 0xFFFFFFFFu }  // AMD64
};

static const ArchDefinition g_coff_arch_def = {
    eArchTypeCOFF,
    llvm::array_lengthof(g_coff_arch_entries),
    g_coff_arch_entries,
    "pe-coff",
};

//===----------------------------------------------------------------------===//
// Table of all ArchDefinitions
static const ArchDefinition *g_arch_definitions[] = {
    &g_macho_arch_def,
    &g_elf_arch_def,
    &g_coff_arch_def
};

static const size_t k_num_arch_definitions = llvm::array_lengthof(g_arch_definitions);

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
    for (unsigned int i = 0; i < llvm::array_lengthof(g_core_definitions); ++i)
    {
        if (name.equals_lower(g_core_definitions[i].name))
            return &g_core_definitions[i];
    }
    return NULL;
}

static inline const CoreDefinition *
FindCoreDefinition (ArchSpec::Core core)
{
    if (core >= 0 && core < llvm::array_lengthof(g_core_definitions))
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
    m_byte_order (eByteOrderInvalid),
    m_distribution_id (),
    m_flags (0)
{
}

ArchSpec::ArchSpec (const char *triple_cstr, Platform *platform) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid),
    m_distribution_id (),
    m_flags (0)
{
    if (triple_cstr)
        SetTriple(triple_cstr, platform);
}


ArchSpec::ArchSpec (const char *triple_cstr) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid),
    m_distribution_id (),
    m_flags (0)
{
    if (triple_cstr)
        SetTriple(triple_cstr);
}

ArchSpec::ArchSpec(const llvm::Triple &triple) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid),
    m_distribution_id (),
    m_flags (0)
{
    SetTriple(triple);
}

ArchSpec::ArchSpec (ArchitectureType arch_type, uint32_t cpu, uint32_t subtype) :
    m_triple (),
    m_core (kCore_invalid),
    m_byte_order (eByteOrderInvalid),
    m_distribution_id (),
    m_flags (0)
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
        m_distribution_id = rhs.m_distribution_id;
        m_flags = rhs.m_flags;
    }
    return *this;
}

void
ArchSpec::Clear()
{
    m_triple = llvm::Triple();
    m_core = kCore_invalid;
    m_byte_order = eByteOrderInvalid;
    m_distribution_id.Clear ();
    m_flags = 0;
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

uint32_t
ArchSpec::GetDataByteSize () const
{
    switch (m_core)
    {
    case eCore_kalimba3:
        return 4;        
    case eCore_kalimba4:
        return 1;        
    case eCore_kalimba5:
        return 4;
    default:        
        return 1;        
    }
    return 1;
}

uint32_t
ArchSpec::GetCodeByteSize () const
{
    switch (m_core)
    {
    case eCore_kalimba3:
        return 4;        
    case eCore_kalimba4:
        return 1;        
    case eCore_kalimba5:
        return 1;        
    default:        
        return 1;        
    }
    return 1;
}

llvm::Triple::ArchType
ArchSpec::GetMachine () const
{
    const CoreDefinition *core_def = FindCoreDefinition (m_core);
    if (core_def)
        return core_def->machine;

    return llvm::Triple::UnknownArch;
}

const ConstString&
ArchSpec::GetDistributionId () const
{
    return m_distribution_id;
}

void
ArchSpec::SetDistributionId (const char* distribution_id)
{
    m_distribution_id.SetCString (distribution_id);
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

bool
ArchSpec::CharIsSignedByDefault () const
{
    switch (m_triple.getArch()) {
    default:
        return true;

    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
        return m_triple.isOSDarwin() || m_triple.isOSWindows();

    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
        return m_triple.isOSDarwin();

    case llvm::Triple::ppc64le:
    case llvm::Triple::systemz:
    case llvm::Triple::xcore:
        return false;
    }
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
        uint32_t cpu = (uint32_t)::strtoul (triple_cstr, &end, 0);
        if (errno == 0 && cpu != 0 && end && ((*end == '-') || (*end == '.')))
        {
            errno = 0;
            uint32_t sub = (uint32_t)::strtoul (end + 1, &end, 0);
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
                            dash_pos = vendor_os.find('-', vendor_start_pos);
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
                *this = HostInfo::GetArchitecture(HostInfo::eArchKind32);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT_64BIT))
                *this = HostInfo::GetArchitecture(HostInfo::eArchKind64);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT))
                *this = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
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
                *this = HostInfo::GetArchitecture(HostInfo::eArchKind32);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT_64BIT))
                *this = HostInfo::GetArchitecture(HostInfo::eArchKind64);
            else if (triple_stref.equals (LLDB_ARCH_DEFAULT))
                *this = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
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
                    if (platform->IsCompatibleArchitecture (raw_arch, false, &compatible_arch))
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

void
ArchSpec::MergeFrom(const ArchSpec &other)
{
    if (GetTriple().getVendor() == llvm::Triple::UnknownVendor && !TripleVendorWasSpecified())
        GetTriple().setVendor(other.GetTriple().getVendor());
    if (GetTriple().getOS() == llvm::Triple::UnknownOS && !TripleOSWasSpecified())
        GetTriple().setOS(other.GetTriple().getOS());
    if (GetTriple().getArch() == llvm::Triple::UnknownArch)
        GetTriple().setArch(other.GetTriple().getArch());
    if (GetTriple().getEnvironment() == llvm::Triple::UnknownEnvironment)
        GetTriple().setEnvironment(other.GetTriple().getEnvironment());
}

bool
ArchSpec::SetArchitecture (ArchitectureType arch_type, uint32_t cpu, uint32_t sub, uint32_t os)
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
                        case llvm::Triple::aarch64:
                        case llvm::Triple::arm:
                        case llvm::Triple::thumb:
                            m_triple.setOS (llvm::Triple::IOS);
                            break;
                            
                        case llvm::Triple::x86:
                        case llvm::Triple::x86_64:
                            // Don't set the OS for x86_64 or for x86 as we want to leave it as an "unspecified unknown"
                            // which means if we ask for the OS from the llvm::Triple we get back llvm::Triple::UnknownOS, but
                            // if we ask for the string value for the OS it will come back empty (unspecified).
                            // We do this because we now have iOS and MacOSX as the OS values for x86 and x86_64 for
                            // normal desktop and simulator binaries. And if we compare a "x86_64-apple-ios" to a "x86_64-apple-"
                            // triple, it will say it is compatible (because the OS is unspecified in the second one and will match
                            // anything in the first
                            break;

                        default:
                            m_triple.setOS (llvm::Triple::MacOSX);
                            break;
                    }
                }
                else if (arch_type == eArchTypeELF)
                {
                    switch (os)
                    {
                        case llvm::ELF::ELFOSABI_AIX:     m_triple.setOS (llvm::Triple::OSType::AIX);     break;
                        case llvm::ELF::ELFOSABI_FREEBSD: m_triple.setOS (llvm::Triple::OSType::FreeBSD); break;
                        case llvm::ELF::ELFOSABI_GNU:     m_triple.setOS (llvm::Triple::OSType::Linux);   break;
                        case llvm::ELF::ELFOSABI_NETBSD:  m_triple.setOS (llvm::Triple::OSType::NetBSD);  break;
                        case llvm::ELF::ELFOSABI_OPENBSD: m_triple.setOS (llvm::Triple::OSType::OpenBSD); break;
                        case llvm::ELF::ELFOSABI_SOLARIS: m_triple.setOS (llvm::Triple::OSType::Solaris); break;
                    }
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
    return IsEqualTo (rhs, true);
}

bool
ArchSpec::IsCompatibleMatch (const ArchSpec& rhs) const
{
    return IsEqualTo (rhs, false);
}

bool
ArchSpec::IsEqualTo (const ArchSpec& rhs, bool exact_match) const
{
    // explicitly ignoring m_distribution_id in this method.

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
            if (exact_match)
            {
                const bool rhs_vendor_specified = rhs.TripleVendorWasSpecified();
                const bool lhs_vendor_specified = TripleVendorWasSpecified();
                // Both architectures had the vendor specified, so if they aren't
                // equal then we return false
                if (rhs_vendor_specified && lhs_vendor_specified)
                    return false;
            }
            
            // Only fail if both vendor types are not unknown
            if (lhs_triple_vendor != llvm::Triple::UnknownVendor &&
                rhs_triple_vendor != llvm::Triple::UnknownVendor)
                return false;
        }
        
        const llvm::Triple::OSType lhs_triple_os = lhs_triple.getOS();
        const llvm::Triple::OSType rhs_triple_os = rhs_triple.getOS();
        if (lhs_triple_os != rhs_triple_os)
        {
            if (exact_match)
            {
                const bool rhs_os_specified = rhs.TripleOSWasSpecified();
                const bool lhs_os_specified = TripleOSWasSpecified();
                // Both architectures had the OS specified, so if they aren't
                // equal then we return false
                if (rhs_os_specified && lhs_os_specified)
                    return false;
            }
            
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

    case ArchSpec::eCore_arm_generic:
        if (enforce_exact_match)
            break;
        // Fall through to case below
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

    case ArchSpec::kCore_x86_64_any:
        if ((core2 >= ArchSpec::kCore_x86_64_first && core2 <= ArchSpec::kCore_x86_64_last) || (core2 == ArchSpec::kCore_x86_64_any))
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

    case ArchSpec::eCore_arm_armv6m:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_generic)
                return true;
            try_inverse = false;
            if (core2 == ArchSpec::eCore_arm_armv7)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv6m)
                return true;
        }
        break;

    case ArchSpec::kCore_hexagon_any:
        if ((core2 >= ArchSpec::kCore_hexagon_first && core2 <= ArchSpec::kCore_hexagon_last) || (core2 == ArchSpec::kCore_hexagon_any))
            return true;
        break;

    case ArchSpec::eCore_arm_armv7em:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_generic)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv7m)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv6m)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv7)
                return true;
            try_inverse = true;
        }
        break;

    case ArchSpec::eCore_arm_armv7m:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_generic)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv6m)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv7)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv7em)
                return true;
            try_inverse = true;
        }
        break;

    case ArchSpec::eCore_arm_armv7f:
    case ArchSpec::eCore_arm_armv7k:
    case ArchSpec::eCore_arm_armv7s:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_generic)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv7)
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_x86_64_x86_64h:
        if (!enforce_exact_match)
        {
            try_inverse = false;
            if (core2 == ArchSpec::eCore_x86_64_x86_64)
                return true;
        }
        break;

    case ArchSpec::eCore_arm_armv8:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_arm64)
                return true;
            if (core2 == ArchSpec::eCore_arm_aarch64)
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_arm_aarch64:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_arm64)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv8)
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_arm_arm64:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_arm_aarch64)
                return true;
            if (core2 == ArchSpec::eCore_arm_armv8)
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_mips32:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32_first && core2 <= ArchSpec::kCore_mips32_last)
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_mips32el:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= ArchSpec::kCore_mips32el_last)
                return true;
            try_inverse = false;
        }

    case ArchSpec::eCore_mips64:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32_first && core2 <= ArchSpec::kCore_mips32_last)
                return true;
            if (core2 >= ArchSpec::kCore_mips64_first && core2 <= ArchSpec::kCore_mips64_last)
                return true;
            try_inverse = false;
        }

    case ArchSpec::eCore_mips64el:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= ArchSpec::kCore_mips32el_last)
                return true;
            if (core2 >= ArchSpec::kCore_mips64el_first && core2 <= ArchSpec::kCore_mips64el_last)
                return true;
            try_inverse = false;
        }

    case ArchSpec::eCore_mips64r2:
    case ArchSpec::eCore_mips64r3:
    case ArchSpec::eCore_mips64r5:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32_first && core2 <= (core1 - 10))
                return true;
            if (core2 >= ArchSpec::kCore_mips64_first && core2 <= (core1 - 1))
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_mips64r2el:
    case ArchSpec::eCore_mips64r3el:
    case ArchSpec::eCore_mips64r5el:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= (core1 - 10))
                return true;
            if (core2 >= ArchSpec::kCore_mips64el_first && core2 <= (core1 - 1))
                return true;
            try_inverse = false;
        }
        break;

    case ArchSpec::eCore_mips32r2:
    case ArchSpec::eCore_mips32r3:
    case ArchSpec::eCore_mips32r5:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32_first && core2 <= core1)
                return true;
        }
        break;

    case ArchSpec::eCore_mips32r2el:
    case ArchSpec::eCore_mips32r3el:
    case ArchSpec::eCore_mips32r5el:
        if (!enforce_exact_match)
        {
            if (core2 >= ArchSpec::kCore_mips32el_first && core2 <= core1)
                return true;
        }
        break;

    case ArchSpec::eCore_mips32r6:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_mips32 || core2 == ArchSpec::eCore_mips32r6)
                return true;
        }
        break;

    case ArchSpec::eCore_mips32r6el:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_mips32el || core2 == ArchSpec::eCore_mips32r6el)
                return true;
                return true;
        }
        break;

    case ArchSpec::eCore_mips64r6:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_mips32 || core2 == ArchSpec::eCore_mips32r6)
                return true;
            if (core2 == ArchSpec::eCore_mips64 || core2 == ArchSpec::eCore_mips64r6)
                return true;
        }
        break;

    case ArchSpec::eCore_mips64r6el:
        if (!enforce_exact_match)
        {
            if (core2 == ArchSpec::eCore_mips32el || core2 == ArchSpec::eCore_mips32r6el)
                return true;
            if (core2 == ArchSpec::eCore_mips64el || core2 == ArchSpec::eCore_mips64r6el)
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
lldb_private::operator<(const ArchSpec& lhs, const ArchSpec& rhs)
{
    const ArchSpec::Core lhs_core = lhs.GetCore ();
    const ArchSpec::Core rhs_core = rhs.GetCore ();
    return lhs_core < rhs_core;
}

static void
StopInfoOverrideCallbackTypeARM(lldb_private::Thread &thread)
{
    // We need to check if we are stopped in Thumb mode in a IT instruction
    // and detect if the condition doesn't pass. If this is the case it means
    // we won't actually execute this instruction. If this happens we need to
    // clear the stop reason to no thread plans think we are stopped for a
    // reason and the plans should keep going.
    //
    // We do this because when single stepping many ARM processes, debuggers
    // often use the BVR/BCR registers that says "stop when the PC is not
    // equal to its current value". This method of stepping means we can end
    // up stopping on instructions inside an if/then block that wouldn't get
    // executed. By fixing this we can stop the debugger from seeming like
    // you stepped through both the "if" _and_ the "else" clause when source
    // level stepping because the debugger stops regardless due to the BVR/BCR
    // triggering a stop.
    //
    // It also means we can set breakpoints on instructions inside an an
    // if/then block and correctly skip them if we use the BKPT instruction.
    // The ARM and Thumb BKPT instructions are unconditional even when executed
    // in a Thumb IT block.
    //
    // If your debugger inserts software traps in ARM/Thumb code, it will
    // need to use 16 and 32 bit instruction for 16 and 32 bit thumb
    // instructions respectively. If your debugger inserts a 16 bit thumb
    // trap on top of a 32 bit thumb instruction for an opcode that is inside
    // an if/then, it will change the it/then to conditionally execute your
    // 16 bit trap and then cause your program to crash if it executes the
    // trailing 16 bits (the second half of the 32 bit thumb instruction you
    // partially overwrote).

    RegisterContextSP reg_ctx_sp (thread.GetRegisterContext());
    if (reg_ctx_sp)
    {
        const uint32_t cpsr = reg_ctx_sp->GetFlags(0);
        if (cpsr != 0)
        {
            // Read the J and T bits to get the ISETSTATE
            const uint32_t J = Bit32(cpsr, 24);
            const uint32_t T = Bit32(cpsr, 5);
            const uint32_t ISETSTATE = J << 1 | T;
            if (ISETSTATE == 0)
            {
                // NOTE: I am pretty sure we want to enable the code below
                // that detects when we stop on an instruction in ARM mode
                // that is conditional and the condition doesn't pass. This
                // can happen if you set a breakpoint on an instruction that
                // is conditional. We currently will _always_ stop on the
                // instruction which is bad. You can also run into this while
                // single stepping and you could appear to run code in the "if"
                // and in the "else" clause because it would stop at all of the
                // conditional instructions in both.
                // In such cases, we really don't want to stop at this location.
                // I will check with the lldb-dev list first before I enable this.
#if 0
                // ARM mode: check for condition on intsruction
                const addr_t pc = reg_ctx_sp->GetPC();
                Error error;
                // If we fail to read the opcode we will get UINT64_MAX as the
                // result in "opcode" which we can use to detect if we read a
                // valid opcode.
                const uint64_t opcode = thread.GetProcess()->ReadUnsignedIntegerFromMemory(pc, 4, UINT64_MAX, error);
                if (opcode <= UINT32_MAX)
                {
                    const uint32_t condition = Bits32((uint32_t)opcode, 31, 28);
                    if (ARMConditionPassed(condition, cpsr) == false)
                    {
                        // We ARE stopped on an ARM instruction whose condition doesn't
                        // pass so this instruction won't get executed.
                        // Regardless of why it stopped, we need to clear the stop info
                        thread.SetStopInfo (StopInfoSP());
                    }
                }
#endif
            }
            else if (ISETSTATE == 1)
            {
                // Thumb mode
                const uint32_t ITSTATE = Bits32 (cpsr, 15, 10) << 2 | Bits32 (cpsr, 26, 25);
                if (ITSTATE != 0)
                {
                    const uint32_t condition = Bits32(ITSTATE, 7, 4);
                    if (ARMConditionPassed(condition, cpsr) == false)
                    {
                        // We ARE stopped in a Thumb IT instruction on an instruction whose
                        // condition doesn't pass so this instruction won't get executed.
                        // Regardless of why it stopped, we need to clear the stop info
                        thread.SetStopInfo (StopInfoSP());
                    }
                }
            }
        }
    }
}

ArchSpec::StopInfoOverrideCallbackType
ArchSpec::GetStopInfoOverrideCallback () const
{
    const llvm::Triple::ArchType machine = GetMachine();
    if (machine == llvm::Triple::arm)
        return StopInfoOverrideCallbackTypeARM;
    return NULL;
}
