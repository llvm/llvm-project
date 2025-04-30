#!/bin/sh

OUTDIR=$1
shift

# Create libc.texinfo from the chapter files.

trap "rm -f ${OUTDIR}*.$$; exit 1" 1 2 15

exec 3>${OUTDIR}incl.$$ 4>${OUTDIR}smenu.$$ 5>${OUTDIR}lmenu.$$

build_menu () {
  while IFS=: read file node; do
    echo "@include $file" >&3
    echo "* $node:: `sed -n 's/^@c %MENU% //p' $file`" >&4
    $AWK 'BEGIN { do_menu = 0 }
	  /^@node / { sub(/^@node /, ""); sub(/,.*$/, ""); node = $0 }
	  /^@menu/ { printf "\n%s\n\n", node; do_menu = 1; next }
	  /^@end menu/ { do_menu = 0 }
	  do_menu { print }' $file >&5
  done
}

collect_nodes () {
  egrep '^(@c )?@node.*Top' "$@" /dev/null | cut -d, -f-2 |
  sed 's/@c //; s/, /:/; s/:@node /:/; s/ /_/g; s/:/ /g' |
  $AWK '{ file[$2] = $1; nnode[$2] = $3 }
	END  { for (x in file)
		 if (file[x] != "")
		   print file[x] ":" x, file[nnode[x]] ":" nnode[x] }' |
  $AWK -f tsort.awk | sed 's/_/ /g'
}

collect_nodes $1 | build_menu

{ echo; echo 'Appendices'; echo; } >&4

collect_nodes $2 | build_menu

exec 3>&- 4>&- 5>&-

mv -f ${OUTDIR}incl.$$ ${OUTDIR}chapters.texi

{
 echo '@menu'
 $AWK -F: '
  /^\*/ {
    printf("%-32s", $1 "::");
    x = split($3, word, " ");
    hpos = 34;
    for (i = 1; i <= x; i++) {
      hpos += length(word[i]) + 1;
      if (hpos > 78) {
	printf("\n%34s", "");
	hpos = 35 + length(word[i]);
      }
      printf(" %s", word[i]);
    }
    print ".";
  }

  !/^\*/ { print; }
 ' ${OUTDIR}smenu.$$
 cat <<EOF
* Free Manuals::		 Free Software Needs Free Documentation.
* Copying::                      The GNU Lesser General Public License says
                                  how you can copy and share the GNU C Library.
* Documentation License::        This manual is under the GNU Free
                                  Documentation License.

Indices

* Concept Index::                Index of concepts and names.
* Type Index::                   Index of types and type qualifiers.
* Function Index::               Index of functions and function-like macros.
* Variable Index::               Index of variables and variable-like macros.
* File Index::                   Index of programs and files.

 @detailmenu
 --- The Detailed Node Listing ---
EOF
 cat ${OUTDIR}lmenu.$$
 echo '@end detailmenu'
 echo '@end menu'; } >${OUTDIR}top-menu.texi.$$
mv -f ${OUTDIR}top-menu.texi.$$ ${OUTDIR}top-menu.texi

rm -f ${OUTDIR}*.$$
