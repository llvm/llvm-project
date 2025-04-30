#!/bin/sh
bindir=$1

VERSION=1.0

egrep -h @deftypefu?nx? *.texi ../linuxthreads/*.texi |
sed -e 's/@deftypefunx*[[:space:]]*\({[^{]*}\|[[:alnum:]_]*\)[[:space:]]*\([[:alnum:]_]*\).*/\2/' -e 's/@deftypefn {[^}]*function}*[[:space:]]*\({[^{]*}\|[[:alnum:]_]*\)[[:space:]]*\([[:alnum:]_]*\).*/\2/' -e '/^@/d' |
sed -e '/^obstack_/d' -e '/^\([lf]\|\)stat\(\|64\)$/d' -e '/^mknod$/d' |
sed -e '/^signbit$/d' -e '/^sigsetjmp$/d' |
sed -e '/^pthread_cleanup/d' -e '/^IFTODT$/d' -e '/^DTTOIF$/d' |
sed -e '/^__fwriting$/d' -e '/^__fwritable$/d' -e '/^__fsetlocking$/d' |
sed -e '/^__freading$/d' -e '/^__freadable$/d' -e '/^__fpurge$/d' |
sed -e '/^__fpending$/d' -e '/^__flbf$/d' -e '/^__fbufsize$/d' |
sed -e '/^alloca$/d' |
sort -u > DOCUMENTED

nm --extern --define $bindir/libc.so $bindir/math/libm.so $bindir/rt/librt.so $bindir/linuxthreads/libpthread.so $bindir/dlfcn/libdl.so $bindir/crypt/libcrypt.so $bindir/login/libutil.so |
egrep " [TW] ([[:alpha:]]|_[[:alpha:]])" |
sed 's/\(@.*\)//' |
cut -b 12- |
sed -e '/^_IO/d' -e '/^_dl/d' -e '/^_pthread/d' -e '/^_obstack/d' |
sed -e '/^_argp/d' -e '/^_authenticate$/d' -e '/^_environ$/d' |
sed -e '/^_errno$/d' -e '/^_h_errno$/d' -e '/^_longjmp$/d' |
sed -e '/^_mcleanup$/d' -e '/^_rpc_dtablesize$/d' -e '/^_seterr_reply$/d' |
sed -e '/^_nss/d' -e '/^_setjmp$/d' |
sort -u > AVAILABLE

cat <<EOF
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<html>
  <head>
    <title>Undocumented glibc functions</title>
  </head>

  <body>
    <center><h1>Undocumented <tt>glibc</tt> functions</h1></center>

    <p>The following table includes names of the function in glibc
    which are not yet documented in the manual.  This list is
    automatically created and therefore might contain errors.  Please
    check the latest manual (available from the CVS archive) before
    starting to work.  It might also be good to let me know in
    advanace on which functions you intend to work to avoid
    duplication.</p>

    <p>A few comments:</p>

    <ul>
      <li>Some functions in the list are much less important than
      others.  Please prioritize.</li>

      <li>Similarly for the LFS functions (those ending in 64).</li>
    </ul>

    <p>The function sombody already volunteered to document are marked
    with a reference to the person.</p>

    <center><table>
EOF

n=0
diff -y --width=60 --suppress-common-lines DOCUMENTED AVAILABLE |
expand | cut -b 33- | sed '/^[[:space:]]*$/d' |
while read name; do
  line="$line
<td><tt>$name</tt></td>"
  n=$(expr $n + 1)
  if [ $n -eq 4 ]; then
    echo "<tr>
$line
</tr>"
    line=""
    n=0
  fi
done
if [ $n -gt 0 ]; then
  if [ $n -eq 1 ]; then
    line="$line
<td></td>"
  fi
  if [ $n -eq 2 ]; then
    line="$line
<td></td>"
  fi
  if [ $n -eq 3 ]; then
    line="$line
<td></td>"
  fi
  echo "<tr>
$line
</tr>"
fi

cat <<EOF
    </table></center>

    <hr>
    <address><a href="mailto:drepper@redhat.com">Ulrich Drepper</a></address>
Generated on $(date) with documented.sh version $VERSION
  </body>
</html>
EOF
