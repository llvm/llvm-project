#!/bin/sh

FILE=$1
BIN_DIR=$2
if [ ! -f $FILE ]; then
  echo "ERROR: Not a file: $FILE"
  exit 3
fi
ret=0

LLVM_NM="$BIN_DIR/llvm-nm"
if [ ! -x $LLVM_NM ]; then
  echo "ERROR: Disassembler '$LLVM_NM' is not executable"
  exit 3
fi

TMP_FILE=$(mktemp)

# Check for external functions
"$LLVM_NM" -u "$FILE" > "$TMP_FILE"
COUNT=$(wc -l < "$TMP_FILE")

if [ "$COUNT" -ne "0" ]; then
  echo "ERROR: $COUNT unresolved external functions detected in $FILE"
  cat $TMP_FILE
  ret=1
else
  echo "File $FILE is OK"
fi
exit $ret
