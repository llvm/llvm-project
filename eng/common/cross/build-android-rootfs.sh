#!/usr/bin/env bash
set -e
__NDK_Version=r21

usage()
{
    echo "Creates a toolchain and sysroot used for cross-compiling for Android."
    echo
    echo "Usage: $0 [BuildArch] [ApiLevel] [--ndk NDKVersion]"
    echo
    echo "BuildArch is the target architecture of Android. Currently only arm64 is supported."
    echo "ApiLevel is the target Android API level. API levels usually match to Android releases. See https://source.android.com/source/build-numbers.html"
    echo "NDKVersion is the version of Android NDK. The default is r21. See https://developer.android.com/ndk/downloads/revision_history"
    echo
    echo "By default, the toolchain and sysroot will be generated in cross/android-rootfs/toolchain/[BuildArch]. You can change this behavior"
    echo "by setting the TOOLCHAIN_DIR environment variable"
    echo
    echo "By default, the NDK will be downloaded into the cross/android-rootfs/android-ndk-$__NDK_Version directory. If you already have an NDK installation,"
    echo "you can set the NDK_DIR environment variable to have this script use that installation of the NDK."
    echo "By default, this script will generate a file, android_platform, in the root of the ROOTFS_DIR directory that contains the RID for the supported and tested Android build: android.28-arm64. This file is to replace '/etc/os-release', which is not available for Android."
    exit 1
}

__ApiLevel=28 # The minimum platform for arm64 is API level 21 but the minimum version that support glob(3) is 28. See $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/glob.h
__BuildArch=arm64
__AndroidArch=aarch64
__AndroidToolchain=aarch64-linux-android

while :; do
    if [[ "$#" -le 0 ]]; then
        break
    fi

    i=$1

    lowerI="$(echo $i | tr "[:upper:]" "[:lower:]")"
    case $lowerI in
        -?|-h|--help)
            usage
            exit 1
            ;;
        arm64)
            __BuildArch=arm64
            __AndroidArch=aarch64
            __AndroidToolchain=aarch64-linux-android
            ;;
        arm)
            __BuildArch=arm
            __AndroidArch=arm
            __AndroidToolchain=arm-linux-androideabi
            ;;
        --ndk)
            shift
            __NDK_Version=$1
            ;;
        *[0-9])
            __ApiLevel=$i
            ;;
        *)
            __UnprocessedBuildArgs="$__UnprocessedBuildArgs $i"
            ;;
    esac
    shift
done

if [[ "$__NDK_Version" == "r21" ]] || [[ "$__NDK_Version" == "r22" ]]; then
    __NDK_File_Arch_Spec=-x86_64
    __SysRoot=sysroot
else
    __NDK_File_Arch_Spec=
    __SysRoot=toolchains/llvm/prebuilt/linux-x86_64/sysroot
fi

# Obtain the location of the bash script to figure out where the root of the repo is.
__ScriptBaseDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

__CrossDir="$__ScriptBaseDir/../../../.tools/android-rootfs"

if [[ ! -f "$__CrossDir" ]]; then
    mkdir -p "$__CrossDir"
fi

# Resolve absolute path to avoid `../` in build logs
__CrossDir="$( cd "$__CrossDir" && pwd )"

__NDK_Dir="$__CrossDir/android-ndk-$__NDK_Version"
__lldb_Dir="$__CrossDir/lldb"
__ToolchainDir="$__CrossDir/android-ndk-$__NDK_Version"

if [[ -n "$TOOLCHAIN_DIR" ]]; then
    __ToolchainDir=$TOOLCHAIN_DIR
fi

if [[ -n "$NDK_DIR" ]]; then
    __NDK_Dir=$NDK_DIR
fi

echo "Target API level: $__ApiLevel"
echo "Target architecture: $__BuildArch"
echo "NDK version: $__NDK_Version"
echo "NDK location: $__NDK_Dir"
echo "Target Toolchain location: $__ToolchainDir"

# Download the NDK if required
if [ ! -d $__NDK_Dir ]; then
    echo Downloading the NDK into $__NDK_Dir
    mkdir -p $__NDK_Dir
    wget -q --progress=bar:force:noscroll --show-progress https://dl.google.com/android/repository/android-ndk-$__NDK_Version-linux$__NDK_File_Arch_Spec.zip -O $__CrossDir/android-ndk-$__NDK_Version-linux.zip
    unzip -q $__CrossDir/android-ndk-$__NDK_Version-linux.zip -d $__CrossDir
fi

if [ ! -d $__lldb_Dir ]; then
    mkdir -p $__lldb_Dir
    echo Downloading LLDB into $__lldb_Dir
    wget -q --progress=bar:force:noscroll --show-progress https://dl.google.com/android/repository/lldb-2.3.3614996-linux-x86_64.zip -O $__CrossDir/lldb-2.3.3614996-linux-x86_64.zip
    unzip -q $__CrossDir/lldb-2.3.3614996-linux-x86_64.zip -d $__lldb_Dir
fi

echo "Download dependencies..."
__TmpDir=$__CrossDir/tmp/$__BuildArch/
mkdir -p "$__TmpDir"

# combined dependencies for coreclr, installer and libraries
__AndroidPackages="libicu"
__AndroidPackages+=" libandroid-glob"
__AndroidPackages+=" liblzma"
__AndroidPackages+=" krb5"
__AndroidPackages+=" openssl"

for path in $(wget -qO- https://packages.termux.dev/termux-main-21/dists/stable/main/binary-$__AndroidArch/Packages |\
    grep -A15 "Package: \(${__AndroidPackages// /\\|}\)" | grep -v "static\|tool" | grep Filename); do

    if [[ "$path" != "Filename:" ]]; then
        echo "Working on: $path"
        wget -qO- https://packages.termux.dev/termux-main-21/$path | dpkg -x - "$__TmpDir"
    fi
done

cp -R "$__TmpDir/data/data/com.termux/files/usr/"* "$__ToolchainDir/$__SysRoot/usr/"

# Generate platform file for build.sh script to assign to __DistroRid
echo "Generating platform file..."
echo "RID=android.${__ApiLevel}-${__BuildArch}" > $__ToolchainDir/$__SysRoot/android_platform

echo "Now to build coreclr, libraries and host; run:"
echo ROOTFS_DIR=$(realpath $__ToolchainDir/$__SysRoot) ./build.sh clr+libs+host --cross --arch $__BuildArch
