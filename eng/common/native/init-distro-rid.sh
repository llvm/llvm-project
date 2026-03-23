#!/bin/sh

# getNonPortableDistroRid
#
# Input:
#   targetOs: (str)
#   targetArch: (str)
#   rootfsDir: (str)
#
# Return:
#   non-portable rid
getNonPortableDistroRid()
{
    targetOs="$1"
    targetArch="$2"
    rootfsDir="$3"
    nonPortableRid=""

    if [ "$targetOs" = "linux" ]; then
        # shellcheck disable=SC1091
        if [ -e "${rootfsDir}/etc/os-release" ]; then
            . "${rootfsDir}/etc/os-release"
            if echo "${VERSION_ID:-}" | grep -qE '^([[:digit:]]|\.)+$'; then
                nonPortableRid="${ID}.${VERSION_ID}-${targetArch}"
            else
                # Rolling release distros either do not set VERSION_ID, set it as blank or
                # set it to non-version looking string (such as TEMPLATE_VERSION_ID on ArchLinux);
                # so omit it here to be consistent with everything else.
                nonPortableRid="${ID}-${targetArch}"
            fi
        elif [ -e "${rootfsDir}/android_platform" ]; then
            # shellcheck disable=SC1091
            . "${rootfsDir}/android_platform"
            nonPortableRid="$RID"
        fi
    fi

    if [ "$targetOs" = "freebsd" ]; then
        # $rootfsDir can be empty. freebsd-version is a shell script and should always work.
        __freebsd_major_version=$("$rootfsDir"/bin/freebsd-version | cut -d'.' -f1)
        nonPortableRid="freebsd.$__freebsd_major_version-${targetArch}"
    elif [ "$targetOs" = "openbsd" ]; then
        nonPortableRid="openbsd.$(uname -r)-${targetArch}"
    elif command -v getprop >/dev/null && getprop ro.product.system.model | grep -qi android; then
        __android_sdk_version=$(getprop ro.build.version.sdk)
        nonPortableRid="android.$__android_sdk_version-${targetArch}"
    elif [ "$targetOs" = "illumos" ]; then
        __uname_version=$(uname -v)
        nonPortableRid="illumos-${targetArch}"
    elif [ "$targetOs" = "solaris" ]; then
        __uname_version=$(uname -v)
        __solaris_major_version=$(echo "$__uname_version" | cut -d'.' -f1)
        nonPortableRid="solaris.$__solaris_major_version-${targetArch}"
    elif [ "$targetOs" = "haiku" ]; then
        __uname_release="$(uname -r)"
        nonPortableRid=haiku.r"$__uname_release"-"$targetArch"
    fi

    echo "$nonPortableRid" | tr '[:upper:]' '[:lower:]'
}

# initDistroRidGlobal
#
# Input:
#   os: (str)
#   arch: (str)
#   rootfsDir?: (nullable:string)
#
# Return:
#   None
#
# Notes:
#   It is important to note that the function does not return anything, but it
#   exports the following variables on success:
#     __DistroRid   : Non-portable rid of the target platform.
#     __PortableTargetOS  : OS-part of the portable rid that corresponds to the target platform.
initDistroRidGlobal()
{
    targetOs="$1"
    targetArch="$2"
    rootfsDir=""
    if [ $# -ge 3 ]; then
        rootfsDir="$3"
    fi

    if [ -n "${rootfsDir}" ]; then
        # We may have a cross build. Check for the existence of the rootfsDir
        if [ ! -e "${rootfsDir}" ]; then
            echo "Error: rootfsDir has been passed, but the location is not valid."
            exit 1
        fi
    fi

    __DistroRid=$(getNonPortableDistroRid "${targetOs}" "${targetArch}" "${rootfsDir}")

    if [ -z "${__PortableTargetOS:-}" ]; then
        __PortableTargetOS="$targetOs"

        STRINGS="$(command -v strings || true)"
        if [ -z "$STRINGS" ]; then
            STRINGS="$(command -v llvm-strings || true)"
        fi

        # Check for musl-based distros (e.g. Alpine Linux, Void Linux).
        if "${rootfsDir}/usr/bin/ldd" --version 2>&1 | grep -q musl ||
                ( [ -n "$STRINGS" ] && "$STRINGS" "${rootfsDir}/usr/bin/ldd" 2>&1 | grep -q musl ); then
            __PortableTargetOS="linux-musl"
        fi
    fi

    export __DistroRid __PortableTargetOS
}
