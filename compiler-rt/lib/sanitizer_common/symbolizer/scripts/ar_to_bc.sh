#!/usr/bin/env bash

function usage() {
  echo "Usage: $0 INPUT... OUTPUT"
  exit 1
}

if [ "$#" -le 1 ]; then
  usage
fi

[[ $AR == /* ]] || AR=$PWD/$AR
[[ $LINK == /* ]] || LINK=$PWD/$LINK

INPUTS=
OUTPUT=
for ARG in $@; do
  INPUTS="$INPUTS $OUTPUT"
  OUTPUT=$(readlink -f $ARG)
done

echo Inputs: $INPUTS
echo Output: $OUTPUT

SCRATCH_DIR=$(mktemp -d)
ln -s $INPUTS $SCRATCH_DIR/

pushd $SCRATCH_DIR

for INPUT in *; do
  for OBJ in $($AR t $INPUT); do
    $AR x $INPUT $OBJ
    if [[ $(file $OBJ) =~ 'LLVM IR bitcode' ]]; then
      mv -f $OBJ $(basename $INPUT).$OBJ
    else
      # Skip $OBJ which may come from an assembly file (e.g. Support/BLAKE3/*.S).
      rm -f $OBJ
    fi
  done
done

$LINK *.o -o $OUTPUT

rm -rf $SCRATCH_DIR
