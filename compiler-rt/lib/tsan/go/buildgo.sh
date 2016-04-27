#!/bin/sh

set -e

SRCS="
	tsan_go.cc
	../rtl/tsan_clock.cc
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
	../../sanitizer_common/sanitizer_allocator.cc
	../../sanitizer_common/sanitizer_common.cc
	../../sanitizer_common/sanitizer_common_libcdep.cc
	../../sanitizer_common/sanitizer_deadlock_detector2.cc
	../../sanitizer_common/sanitizer_flag_parser.cc
	../../sanitizer_common/sanitizer_flags.cc
	../../sanitizer_common/sanitizer_libc.cc
	../../sanitizer_common/sanitizer_persistent_allocator.cc
	../../sanitizer_common/sanitizer_printf.cc
	../../sanitizer_common/sanitizer_suppressions.cc
	../../sanitizer_common/sanitizer_thread_registry.cc
	../../sanitizer_common/sanitizer_stackdepot.cc
	../../sanitizer_common/sanitizer_stacktrace.cc
	../../sanitizer_common/sanitizer_symbolizer.cc
"

if [ "`uname -a | grep Linux`" != "" ]; then
	SUFFIX="linux_amd64"
	OSCFLAGS="-fPIC -ffreestanding -Wno-maybe-uninitialized -Wno-unused-const-variable -Werror -Wno-unknown-warning-option"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cc
		../../sanitizer_common/sanitizer_posix_libcdep.cc
		../../sanitizer_common/sanitizer_procmaps_common.cc
		../../sanitizer_common/sanitizer_procmaps_linux.cc
		../../sanitizer_common/sanitizer_linux.cc
		../../sanitizer_common/sanitizer_linux_libcdep.cc
		../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cc
	"
elif [ "`uname -a | grep FreeBSD`" != "" ]; then
	SUFFIX="freebsd_amd64"
	OSCFLAGS="-fno-strict-aliasing -fPIC -Werror"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_linux.cc
		../../sanitizer_common/sanitizer_posix.cc
		../../sanitizer_common/sanitizer_posix_libcdep.cc
		../../sanitizer_common/sanitizer_procmaps_common.cc
		../../sanitizer_common/sanitizer_procmaps_freebsd.cc
		../../sanitizer_common/sanitizer_linux.cc
		../../sanitizer_common/sanitizer_linux_libcdep.cc
		../../sanitizer_common/sanitizer_stoptheworld_linux_libcdep.cc
	"
elif [ "`uname -a | grep Darwin`" != "" ]; then
	SUFFIX="darwin_amd64"
	OSCFLAGS="-fPIC -Wno-unused-const-variable -Wno-unknown-warning-option"
	OSLDFLAGS="-lpthread -fPIC -fpie"
	SRCS="
		$SRCS
		../rtl/tsan_platform_mac.cc
		../../sanitizer_common/sanitizer_mac.cc
		../../sanitizer_common/sanitizer_posix.cc
		../../sanitizer_common/sanitizer_posix_libcdep.cc
		../../sanitizer_common/sanitizer_procmaps_mac.cc
	"
elif [ "`uname -a | grep MINGW`" != "" ]; then
	SUFFIX="windows_amd64"
	OSCFLAGS="-Wno-error=attributes -Wno-attributes -Wno-unused-const-variable -Wno-unknown-warning-option"
	OSLDFLAGS=""
	SRCS="
		$SRCS
		../rtl/tsan_platform_windows.cc
		../../sanitizer_common/sanitizer_win.cc
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

FLAGS=" -I../rtl -I../.. -I../../sanitizer_common -I../../../include -std=c++11 -m64 -Wall -fno-exceptions -fno-rtti -DSANITIZER_GO -DSANITIZER_DEADLOCK_DETECTOR_VERSION=2 $OSCFLAGS"
if [ "$DEBUG" = "" ]; then
	FLAGS="$FLAGS -DSANITIZER_DEBUG=0 -O3 -msse3 -fomit-frame-pointer"
else
	FLAGS="$FLAGS -DSANITIZER_DEBUG=1 -g"
fi

if [ "$SILENT" != "1" ]; then
  echo $CC gotsan.cc -c -o $DIR/race_$SUFFIX.syso $FLAGS $CFLAGS
fi
$CC $DIR/gotsan.cc -c -o $DIR/race_$SUFFIX.syso $FLAGS $CFLAGS

$CC $OSCFLAGS test.c $DIR/race_$SUFFIX.syso -m64 -g -o $DIR/test $OSLDFLAGS

export GORACE="exitcode=0 atexit_sleep_ms=0"
if [ "$SILENT" != "1" ]; then
  $DIR/test
else
  $DIR/test 2>/dev/null
fi
