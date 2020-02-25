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

case $# in
  (1) ;;
  (2) F18=$2 ;;
  (*) echo "Usage: $(basename $0) <fortran-source> [<f18-executable>]"; exit 1
esac
[[ -z ${F18+x} ]] && die "Path to f18 must be second argument or in F18 environment variable"
[[ ! -f $F18 ]] && die "f18 executable not found: $F18"
case $1 in
  (/*) src=$1 ;;
  (*) src=$(dirname $0)/$1 ;;
esac
USER_OPTIONS=`sed -n 's/^ *! *OPTIONS: *//p' $src`
echo $USER_OPTIONS
temp=`mktemp -d ./tmp.XXXXXX`
[[ $KEEP ]] || trap "rm -rf $temp" EXIT
