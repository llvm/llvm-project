// RUN: %clang -### %s -c -o tmp.o -triple i686-pc-linux-gnu -integrated-as -Wa,--noexecstack 2>&1 | grep "mnoexecstack"
