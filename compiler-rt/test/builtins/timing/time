#!/bin/sh

function test () {
    arch=$1
    file=$2
    name=$3
    ldflags=$4

    if gcc -arch $arch -Os $file $ldflags -DLIBNAME=$name
    then
	if ./a.out
	then
	    rm ./a.out
	else
	    echo "fail"
	fi
    else
	echo "$FILE failed to compile"
    fi
}

INSTALLED=/usr/local/lib/system/libcompiler_rt.a

for ARCH in i386 x86_64; do
	for FILE in $(ls *.c); do
		
		echo "Timing $FILE for $ARCH"

		test $ARCH $FILE libgcc ""
                test $ARCH $FILE tuned ../../darwin_fat/Release/libcompiler_rt.a
                if [ -f "$INSTALLED" ]; then
                    test $ARCH $FILE installed $INSTALLED
		fi

		echo " "
		
	done
done
exit
