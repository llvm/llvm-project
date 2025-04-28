#!/usr/bin/env bash
set -e

if [[ -z "${VERBOSE// }" ]] || [ "$VERBOSE" -ne "$VERBOSE" ] 2>/dev/null; then
    VERBOSE=0
fi

Log()
{
    if [ $VERBOSE -ge 1 ]; then
        echo ${@:2}
    fi
}

Inform()
{
    Log 1 -e "\x1B[0;34m$@\x1B[m"
}

Debug()
{
    Log 2 -e "\x1B[0;32m$@\x1B[m"
}

Error()
{
    >&2 Log 0 -e "\x1B[0;31m$@\x1B[m"
}

Fetch()
{
    URL=$1
    FILE=$2
    PROGRESS=$3
    if [ $VERBOSE -ge 1 ] && [ $PROGRESS ]; then
        CURL_OPT="--progress-bar"
    else
        CURL_OPT="--silent"
    fi
    curl $CURL_OPT $URL > $FILE
}

hash curl 2> /dev/null || { Error "Require 'curl' Aborting."; exit 1; }
hash xmllint 2> /dev/null || { Error "Require 'xmllint' Aborting."; exit 1; }
hash sha256sum 2> /dev/null || { Error "Require 'sha256sum' Aborting."; exit 1; }

TMPDIR=$1
if [ ! -d $TMPDIR ]; then
    TMPDIR=./tizen_tmp
    Debug "Create temporary directory : $TMPDIR"
    mkdir -p $TMPDIR
fi

TIZEN_ARCH=$2

TIZEN_URL=http://download.tizen.org/snapshots/TIZEN/Tizen
BUILD_XML=build.xml
REPOMD_XML=repomd.xml
PRIMARY_XML=primary.xml
TARGET_URL="http://__not_initialized"

Xpath_get()
{
    XPATH_RESULT=''
    XPATH=$1
    XML_FILE=$2
    RESULT=$(xmllint --xpath $XPATH $XML_FILE)
    if [[ -z ${RESULT// } ]]; then
        Error "Can not find target from $XML_FILE"
        Debug "Xpath = $XPATH"
        exit 1
    fi
    XPATH_RESULT=$RESULT
}

fetch_tizen_pkgs_init()
{
    TARGET=$1
    PROFILE=$2
    Debug "Initialize TARGET=$TARGET, PROFILE=$PROFILE"

    TMP_PKG_DIR=$TMPDIR/tizen_${PROFILE}_pkgs
    if [ -d $TMP_PKG_DIR ]; then rm -rf $TMP_PKG_DIR; fi
    mkdir -p $TMP_PKG_DIR

    PKG_URL=$TIZEN_URL/$PROFILE/latest

    BUILD_XML_URL=$PKG_URL/$BUILD_XML
    TMP_BUILD=$TMP_PKG_DIR/$BUILD_XML
    TMP_REPOMD=$TMP_PKG_DIR/$REPOMD_XML
    TMP_PRIMARY=$TMP_PKG_DIR/$PRIMARY_XML
    TMP_PRIMARYGZ=${TMP_PRIMARY}.gz

    Fetch $BUILD_XML_URL $TMP_BUILD

    Debug "fetch $BUILD_XML_URL to $TMP_BUILD"

    TARGET_XPATH="//build/buildtargets/buildtarget[@name=\"$TARGET\"]/repo[@type=\"binary\"]/text()"
    Xpath_get $TARGET_XPATH $TMP_BUILD
    TARGET_PATH=$XPATH_RESULT
    TARGET_URL=$PKG_URL/$TARGET_PATH

    REPOMD_URL=$TARGET_URL/repodata/repomd.xml
    PRIMARY_XPATH='string(//*[local-name()="data"][@type="primary"]/*[local-name()="location"]/@href)'

    Fetch $REPOMD_URL $TMP_REPOMD

    Debug "fetch $REPOMD_URL to $TMP_REPOMD"

    Xpath_get $PRIMARY_XPATH $TMP_REPOMD
    PRIMARY_XML_PATH=$XPATH_RESULT
    PRIMARY_URL=$TARGET_URL/$PRIMARY_XML_PATH

    Fetch $PRIMARY_URL $TMP_PRIMARYGZ

    Debug "fetch $PRIMARY_URL to $TMP_PRIMARYGZ"

    gunzip $TMP_PRIMARYGZ

    Debug "unzip $TMP_PRIMARYGZ to $TMP_PRIMARY"
}

fetch_tizen_pkgs()
{
    ARCH=$1
    PACKAGE_XPATH_TPL='string(//*[local-name()="metadata"]/*[local-name()="package"][*[local-name()="name"][text()="_PKG_"]][*[local-name()="arch"][text()="_ARCH_"]]/*[local-name()="location"]/@href)'

    PACKAGE_CHECKSUM_XPATH_TPL='string(//*[local-name()="metadata"]/*[local-name()="package"][*[local-name()="name"][text()="_PKG_"]][*[local-name()="arch"][text()="_ARCH_"]]/*[local-name()="checksum"]/text())'

    for pkg in ${@:2}
    do
        Inform "Fetching... $pkg"
        XPATH=${PACKAGE_XPATH_TPL/_PKG_/$pkg}
        XPATH=${XPATH/_ARCH_/$ARCH}
        Xpath_get $XPATH $TMP_PRIMARY
        PKG_PATH=$XPATH_RESULT

        XPATH=${PACKAGE_CHECKSUM_XPATH_TPL/_PKG_/$pkg}
        XPATH=${XPATH/_ARCH_/$ARCH}
        Xpath_get $XPATH $TMP_PRIMARY
        CHECKSUM=$XPATH_RESULT

        PKG_URL=$TARGET_URL/$PKG_PATH
        PKG_FILE=$(basename $PKG_PATH)
        PKG_PATH=$TMPDIR/$PKG_FILE

        Debug "Download $PKG_URL to $PKG_PATH"
        Fetch $PKG_URL $PKG_PATH true

        echo "$CHECKSUM $PKG_PATH" | sha256sum -c - > /dev/null
        if [ $? -ne 0 ]; then
            Error "Fail to fetch $PKG_URL to $PKG_PATH"
            Debug "Checksum = $CHECKSUM"
            exit 1
        fi
    done
}

BASE="Tizen-Base"
UNIFIED="Tizen-Unified"

Inform "Initialize ${TIZEN_ARCH} base"
fetch_tizen_pkgs_init standard $BASE
Inform "fetch common packages"
fetch_tizen_pkgs ${TIZEN_ARCH} gcc gcc-devel-static glibc glibc-devel libicu libicu-devel libatomic linux-glibc-devel keyutils keyutils-devel libkeyutils
Inform "fetch coreclr packages"
fetch_tizen_pkgs ${TIZEN_ARCH} libgcc libstdc++ libstdc++-devel libunwind libunwind-devel lttng-ust-devel lttng-ust userspace-rcu-devel userspace-rcu
if [ "$TIZEN_ARCH" != "riscv64" ]; then
    fetch_tizen_pkgs ${TIZEN_ARCH} lldb lldb-devel
fi
Inform "fetch corefx packages"
fetch_tizen_pkgs ${TIZEN_ARCH} libcom_err libcom_err-devel zlib zlib-devel libopenssl11 libopenssl1.1-devel krb5 krb5-devel

Inform "Initialize standard unified"
fetch_tizen_pkgs_init standard $UNIFIED
Inform "fetch corefx packages"
fetch_tizen_pkgs ${TIZEN_ARCH} gssdp gssdp-devel tizen-release

