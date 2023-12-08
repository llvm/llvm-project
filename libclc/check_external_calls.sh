#!/bin/sh

FILE=$1
BIN_DIR=$2
if [ ! -f $FILE ]; then
	echo "ERROR: Not a file: $FILE"
	exit 3
fi
ret=0

DIS="$BIN_DIR/llvm-dis"
if [ ! -x $DIS ]; then
	echo "ERROR: Disassembler '$DIS' is not executable"
	exit 3
fi

TMP_FILE=$(mktemp)

# Check for calls. Calls to llvm intrinsics are OK
$DIS < $FILE | grep ' call ' | grep -v '@llvm' > "$TMP_FILE"
COUNT=$(wc -l < "$TMP_FILE")

if [ "$COUNT" -ne "0" ]; then
	echo "ERROR: $COUNT unresolved calls detected in $FILE"
	cat $TMP_FILE
	ret=1
else
	echo "File $FILE is OK"
fi
exit $ret
