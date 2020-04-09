# Common functionality for test scripts
# Process arguments, expecting source file as 1st; optional path to f18 as 2nd
# Set: $F18 to the path to f18; $temp to an empty temp directory; $src
# to the full path of the single source argument; and $USER_OPTIONS to the
# option list given in the $src file after string "OPTIONS:"

PATH=/usr/bin:/bin

function die {
  echo "$(basename $0): $*" >&2
  exit 1
}
if [[ $# < 3 ]]; then
  echo "Usage: $(basename $0) <fortran-source> <f18-executable> <temp test dir>"
  exit 1
fi

case $1 in
  (/*) src=$1 ;;
  (*) src=$(dirname $0)/$1 ;;
esac
USER_OPTIONS=`sed -n 's/^ *! *OPTIONS: *//p' $src`
echo $USER_OPTIONS
F18=$2
[[ ! -f $F18 ]] && die "f18 executable not found: $F18"
temp=$3
mkdir -p $temp
