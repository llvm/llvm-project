""" Utility module to determine the OS Python running on

    --------------------------------------------------------------------------
    File:         utilsOsType.py

    Overview:       Python module to supply functions and an enumeration to
                    help determine the platform type, bit size and OS currently
                    being used.
    --------------------------------------------------------------------------

"""

# Python modules:
import sys      # Provide system information

# Third party modules:

# In-house modules:

# Instantiations:

# Enumerations:
#-----------------------------------------------------------------------------
# Details:  Class to implement a 'C' style enumeration type.
# Gotchas:  None.
# Authors:  Illya Rudkin 28/11/2013.
# Changes:  None.
#--
if sys.version_info.major >= 3:
    from enum import Enum

    class EnumOsType(Enum):
        Unknown = 0
        Darwin = 1
        FreeBSD = 2
        Linux = 3
        NetBSD = 4
        Windows = 5
        kFreeBSD = 6
else:
    class EnumOsType(object):
        values = ["Unknown",
                  "Darwin",
                  "FreeBSD",
                  "Linux",
                  "NetBSD",
                  "OpenBSD",
                  "Windows",
                  "kFreeBSD"]

        class __metaclass__(type):
            #++----------------------------------------------------------------
            # Details:  Fn acts as an enumeration.
            # Args:     vName - (R) Enumeration to match.
            # Returns:  Int - Matching enumeration/index.
            # Throws:   None.
            #--

            def __getattr__(cls, vName):
                return cls.values.index(vName)

#++---------------------------------------------------------------------------
# Details:  Reverse fast lookup of the values list.
# Args:     vI - (R) Index / enumeration.
# Returns:  Str - text description matching enumeration.
# Throws:   None.
#--
            def name_of(cls, vI):
                return EnumOsType.values[vI]

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#++---------------------------------------------------------------------------
# Details:  Determine what operating system is currently running on.
# Args:     None.
# Returns:  EnumOsType - The OS type being used ATM.
# Throws:   None.
#--


def determine_os_type():
    eOSType = EnumOsType.Unknown

    strOS = sys.platform
    if strOS == "darwin":
        eOSType = EnumOsType.Darwin
    elif strOS.startswith("freebsd"):
        eOSType = EnumOsType.FreeBSD
    elif strOS.startswith("linux"):
        eOSType = EnumOsType.Linux
    elif strOS.startswith("netbsd"):
        eOSType = EnumOsType.NetBSD
    elif strOS.startswith("openbsd"):
        eOSType = EnumOsType.OpenBSD
    elif strOS == "win32":
        eOSType = EnumOsType.Windows
    elif strOS.startswith("gnukfreebsd"):
        eOSType = EnumOsType.kFreeBSD

    return eOSType
