#!/bin/sh


# This script will install the files from a "Debug" or "Release" build
# directory into the developer folder specified.

NUM_EXPECTED_ARGS=2

PROGRAM=`basename $0`

if [ $# -ne $NUM_EXPECTED_ARGS ]; then
	echo This script will install the files from a 'Debug' or 'Release' build directory into the developer folder specified.
	echo "usage: $PROGRAM <BUILD_DIR> <DEVELOPER_DIR>";
	echo "example: $PROGRAM ./Debug /Developer"
	echo "example: $PROGRAM /build/Release /Xcode4"
	exit 1;
fi

BUILD_DIR=$1
DEVELOPER_DIR=$2

if [ -d $BUILD_DIR ]; then
	if [ -d $DEVELOPER_DIR ]; then
		if [ -e "$BUILD_DIR/debugserver" ]; then
			echo Updating "$DEVELOPER_DIR/usr/bin/debugserver"
			sudo rm -rf "$DEVELOPER_DIR/usr/bin/debugserver"
			sudo cp "$BUILD_DIR/debugserver" "$DEVELOPER_DIR/usr/bin/debugserver"
		fi

		if [ -e "$BUILD_DIR/lldb" ]; then
			echo Updating "$DEVELOPER_DIR/usr/bin/lldb"
			sudo rm -rf "$DEVELOPER_DIR/usr/bin/lldb"
			sudo cp "$BUILD_DIR/lldb" "$DEVELOPER_DIR/usr/bin/lldb"
		fi

		if [ -e "$BUILD_DIR/libEnhancedDisassembly.dylib" ]; then
			echo Updating "$DEVELOPER_DIR/usr/lib/libEnhancedDisassembly.dylib"
			sudo rm -rf "$DEVELOPER_DIR/usr/lib/libEnhancedDisassembly.dylib"
			sudo cp "$BUILD_DIR/libEnhancedDisassembly.dylib" "$DEVELOPER_DIR/usr/lib/libEnhancedDisassembly.dylib"
		fi

		if [ -d "$BUILD_DIR/LLDB.framework" ]; then
			echo Updating "$DEVELOPER_DIR/Library/PrivateFrameworks/LLDB.framework"
			sudo rm -rf "$DEVELOPER_DIR/Library/PrivateFrameworks/LLDB.framework"
			sudo cp -r "$BUILD_DIR/LLDB.framework" "$DEVELOPER_DIR/Library/PrivateFrameworks/LLDB.framework"
		elif [ -e "$BUILD_DIR/LLDB.framework" ]; then
			echo BUILD_DIR path to LLDB.framework is not a directory: "$BUILD_DIR/LLDB.framework"
			exit 2;			
		fi
	
	else
		echo DEVELOPER_DIR must be a directory: "$DEVELOPER_DIR"
		exit 3;	
	fi

else
	echo BUILD_DIR must be a directory: "$BUILD_DIR"
	exit 4;	
fi
