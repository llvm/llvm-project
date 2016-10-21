# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Take some valid script with multiline comments
## and check it actually works:
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ".text : { *(.text) }" >> %t.script
# RUN: echo ".keep : { *(.keep) } /*" >> %t.script
# RUN: echo "comment line 1" >> %t.script
# RUN: echo "comment line 2 */" >> %t.script
# RUN: echo ".temp : { *(.temp) } }" >> %t.script
# RUN: ld.lld -shared %t -o %t1 --script %t.script

## Change ":" to "+" at line 2, check that error
## message starts from correct line number:
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ".text + { *(.text) }" >> %t.script
# RUN: echo ".keep : { *(.keep) } /*" >> %t.script
# RUN: echo "comment line 1" >> %t.script
# RUN: echo "comment line 2 */" >> %t.script
# RUN: echo ".temp : { *(.temp) } }" >> %t.script
# RUN: not ld.lld -shared %t -o %t1 --script %t.script 2>&1 | FileCheck -check-prefix=ERR1 %s
# ERR1: line 2:

## Change ":" to "+" at line 3 now, check correct error line number:
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ".text : { *(.text) }" >> %t.script
# RUN: echo ".keep + { *(.keep) } /*" >> %t.script
# RUN: echo "comment line 1" >> %t.script
# RUN: echo "comment line 2 */" >> %t.script
# RUN: echo ".temp : { *(.temp) } }" >> %t.script
# RUN: not ld.lld -shared %t -o %t1 --script %t.script 2>&1 | FileCheck -check-prefix=ERR2 %s
# ERR2: line 3:

## Change ":" to "+" at line 6, after multiline comment,
## check correct error line number:
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ".text : { *(.text) }" >> %t.script
# RUN: echo ".keep : { *(.keep) } /*" >> %t.script
# RUN: echo "comment line 1" >> %t.script
# RUN: echo "comment line 2 */" >> %t.script
# RUN: echo ".temp + { *(.temp) } }" >> %t.script
# RUN: not ld.lld -shared %t -o %t1 --script %t.script 2>&1 | FileCheck -check-prefix=ERR5 %s
# ERR5: line 6:

## Check that text of lines and pointer to 'bad' token are working ok.
# RUN: echo "UNKNOWN_TAG {" > %t.script
# RUN: echo ".text : { *(.text) }" >> %t.script
# RUN: echo ".keep : { *(.keep) }" >> %t.script
# RUN: echo ".temp : { *(.temp) } }" >> %t.script
# RUN: not ld.lld -shared %t -o %t1 --script %t.script 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR6 -strict-whitespace %s
# ERR6:      error: line 1:
# ERR6-NEXT: error: UNKNOWN_TAG {
# ERR6-NEXT: error: ^

## One more check that text of lines and pointer to 'bad' token are working ok.
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ".text : { *(.text) }" >> %t.script
# RUN: echo ".keep : { *(.keep) }" >> %t.script
# RUN: echo "boom .temp : { *(.temp) } }" >> %t.script
# RUN: not ld.lld -shared %t -o %t1 --script %t.script 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR7 -strict-whitespace %s
# ERR7:      error: line 4: malformed number: .temp
# ERR7-NEXT: error: boom .temp : { *(.temp) } }
# ERR7-NEXT: error:      ^
