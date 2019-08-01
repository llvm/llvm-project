#!/bin/sh

set -e

SRCS="
	tsan_go.cc
	../rtl/tsan_clock.cc
	../rtl/tsan_external.cc
	../rtl/tsan_flags.cc
	../rtl/tsan_interface_atomic.cc
	../rtl/tsan_md5.cc
	../rtl/tsan_mutex.cc
	../rtl/tsan_report.cc
	../rtl/tsan_rtl.cc
	../rtl/tsan_rtl_mutex.cc
	../rtl/tsan_rtl_report.cc
	../rtl/tsan_rtl_thread.cc
	../rtl/tsan_rtl_proc.cc
	../rtl/tsan_stack_trace.cc
	../rtl/tsan_stat.cc
	../rtl/tsan_suppressions.cc
	../rtl/tsan_sync.cc
	../../sanitizer_common/sanitizer_allocator.cpp
	../../sanitizer_common/sanitizer_common.cpp
	../../sanitizer_common/sanitizer_common_libcdep.cpp
	../../sanitizer_common/sanitizer_deadlock_detector2.cpp
	../../sanitizer_common/sanitizer_file.cpp
	../../sanitizer_common/sanitizer_flag_parser.cpp
	../../sanitizer_common/sanitizer_flags.cpp
	../../sanitizer_common/sanitizer_libc.cpp
	../../sanitizer_common/sanitizer_persistent_allocator.cpp
	../../sanitizer_common/sanitizer_printf.cpp
	../../sanitizer_common/sanitizer_suppressions.cpp
	../../sanitizer_common/sanitizer_thread_registry.cpp
	../../sanitizer_common/sanitizer_stackdepot.cpp
	../../sanitizer_common/sanitizer_stacktrace.cpp
	../../sanitizer_common/sanitizer_symbolizer.cpp
	../../sanitizer_common/sanitizer_symbolizer_report.cpp
	../../sanitizer_common/sanitizer_termination.cpp
"

if [ "`uname -a | grep Linux`" != "" ]; then
	OSCFLAGS="-fPIC -Wno-maybe-uninitialized"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cpp
		../../sanitizer_common/sanitizer_posix_libcdep.cpp
		../../sanitizer_common/sanitizer_procmaps_common.cpp
		../../sanitizer_common/sanitizer_procmaps_linux.cpp
		../../sanitizer_common/sanitizer_linux.cpp
		../../sanitizer_common/sanitizer_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_stoptheworld_netbsd_libcdep.cpp
		"
	if [ "`uname -a | grep ppc64le`" != "" ]; then
		SUFFIX="linux_ppc64le"
		ARCHCFLAGS="-m64"
	elif [ "`uname -a | grep x86_64`" != "" ]; then
		SUFFIX="linux_amd64"
		ARCHCFLAGS="-m64"
		OSCFLAGS="$OSCFLAGS -ffreestanding -Wno-unused-const-variable -Werror -Wno-unknown-warning-option"
	elif [ "`uname -a | grep aarch64`" != "" ]; then
		SUFFIX="linux_arm64"
		ARCHCFLAGS=""
	fi
elif [ "`uname -a | grep FreeBSD`" != "" ]; then
	SUFFIX="freebsd_amd64"
	OSCFLAGS="-fno-strict-aliasing -fPIC -Werror"
	ARCHCFLAGS="-m64"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cpp
		../../sanitizer_common/sanitizer_posix_libcdep.cpp
		../../sanitizer_common/sanitizer_procmaps_bsd.cpp
		../../sanitizer_common/sanitizer_procmaps_common.cpp
		../../sanitizer_common/sanitizer_linux.cpp
		../../sanitizer_common/sanitizer_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_stoptheworld_netbsd_libcdep.cpp
	"
elif [ "`uname -a | grep NetBSD`" != "" ]; then
	SUFFIX="netbsd_amd64"
	OSCFLAGS="-fno-strict-aliasing -fPIC -Werror"
	ARCHCFLAGS="-m64"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cpp
		../../sanitizer_common/sanitizer_posix_libcdep.cpp
		../../sanitizer_common/sanitizer_procmaps_bsd.cpp
		../../sanitizer_common/sanitizer_procmaps_common.cpp
		../../sanitizer_common/sanitizer_linux.cpp
		../../sanitizer_common/sanitizer_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_netbsd.cpp
		../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cpp
		../../sanitizer_common/sanitizer_stoptheworld_netbsd_libcdep.cpp
	"
elif [ "`uname -a | grep Darwin`" != "" ]; then
	SUFFIX="darwin_amd64"
	OSCFLAGS="-fPIC -Wno-unused-const-variable -Wno-unknown-warning-option -mmacosx-version-min=10.7"
	ARCHCFLAGS="-m64"
	OSLDFLAGS="-lpthread -fPIC -fpie -mmacosx-version-min=10.7"
	SRCS="
		$SRCS
		../rtl/tsan_platform_mac.cc
		../../sanitizer_common/sanitizer_mac.cpp
		../../sanitizer_common/sanitizer_posix.cpp
		../../sanitizer_common/sanitizer_posix_libcdep.cpp
		../../sanitizer_common/sanitizer_procmaps_mac.cpp
	"
elif [ "`uname -a | grep MINGW`" != "" ]; then
	SUFFIX="windows_amd64"
	OSCFLAGS="-Wno-error=attributes -Wno-attributes -Wno-unused-const-variable -Wno-unknown-warning-option"
	ARCHCFLAGS="-m64"
	OSLDFLAGS=""
	SRCS="
		$SRCS
		../rtl/tsan_platform_windows.cc
		../../sanitizer_common/sanitizer_win.cpp
	"
else
	echo Unknown platform
	exit 1
fi

CC=${CC:-gcc}
IN_TMPDIR=${IN_TMPDIR:-0}
SILENT=${SILENT:-0}

if [ $IN_TMPDIR != "0" ]; then
  DIR=$(mktemp -qd /tmp/gotsan.XXXXXXXXXX)
  cleanup() {
    rm -rf $DIR
  }
  trap cleanup EXIT
else
  DIR=.
fi

SRCS="$SRCS $ADD_SRCS"

rm -f $DIR/gotsan.cc
for F in $SRCS; do
	cat $F >> $DIR/gotsan.cc
done

FLAGS=" -I../rtl -I../.. -I../../sanitizer_common -I../../../include -std=c++11 -Wall -fno-exceptions -fno-rtti -DSANITIZER_GO=1 -DSANITIZER_DEADLOCK_DETECTOR_VERSION=2 $OSCFLAGS $ARCHCFLAGS"
if [ "$DEBUG" = "" ]; then
	FLAGS="$FLAGS -DSANITIZER_DEBUG=0 -O3 -fomit-frame-pointer"
	if [ "$SUFFIX" = "linux_ppc64le" ]; then
		FLAGS="$FLAGS -mcpu=power8 -fno-function-sections"
	elif [ "$SUFFIX" = "linux_amd64" ]; then
		FLAGS="$FLAGS -msse3"
	fi
else
	FLAGS="$FLAGS -DSANITIZER_DEBUG=1 -g"
fi

if [ "$SILENT" != "1" ]; then
  echo $CC gotsan.cc -c -o $DIR/race_$SUFFIX.syso $FLAGS $CFLAGS
fi
$CC $DIR/gotsan.cc -c -o $DIR/race_$SUFFIX.syso $FLAGS $CFLAGS

$CC $OSCFLAGS $ARCHCFLAGS test.c $DIR/race_$SUFFIX.syso -g -o $DIR/test $OSLDFLAGS $LDFLAGS

export GORACE="exitcode=0 atexit_sleep_ms=0"
if [ "$SILENT" != "1" ]; then
  $DIR/test
else
  $DIR/test 2>/dev/null
fi
