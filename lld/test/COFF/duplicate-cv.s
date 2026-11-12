# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: cp %t.obj %t.dupl.obj
# RUN: not lld-link /out:%t.exe %t.obj %t.dupl.obj 2>&1 | FileCheck %s

# CHECK: error: duplicate symbol: main
# CHECK-NEXT: >>> defined at file1.cpp:2
# CHECK-NEXT: >>>            {{.*}}.obj
# CHECK-NEXT: >>> defined at {{.*}}.obj

	.cv_file	1 "file1.cpp" "EDA15C78BB573E49E685D8549286F33C" 1
	.cv_file	2 "file2.cpp" "EDA15C78BB573E49E685D8549286F33D" 1

        .section        .text,"xr",one_only,main
.globl main
main:
	.cv_func_id 0
	.cv_loc	0 1 1 0 is_stmt 0
	.cv_loc	0 1 2 0
	retq
.Lfunc_end0:

	.section	.debug$S,"dr",associative,main
	.long	4
	.cv_linetable	0, main, .Lfunc_end0

	.section	.debug$S,"dr"
	.long	4
	.cv_filechecksums
	.cv_stringtable
