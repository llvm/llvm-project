// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang_cl /c -flto -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// DEFAULT: "-o" "cl-outputs.obj"

// RUN: %clang_cl /Fo -### -- %s 2>&1 | FileCheck -check-prefix=FoEMPTY %s
// FoEMPTY:  "-o" "cl-outputs.obj"

// RUN: %clang_cl /Foa -### -- %s 2>&1 | FileCheck -check-prefix=FoNAME %s
// RUN: not %clang_cl /Foa -flto -### -- %s 2>&1 | FileCheck -check-prefix=FoNAME %s
// FoNAME:  "-o" "a.obj"

// RUN: %clang_cl /Foa.ext /Fob.ext -### -- %s 2>&1 | FileCheck -check-prefix=FoNAMEEXT %s
// FoNAMEEXT:  "-o" "b.ext"

// RUN: %clang_cl /Fofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FoDIR %s
// FoDIR:  "-o" "foo.dir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /Fofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FoDIRNAME %s
// FoDIRNAME:  "-o" "foo.dir{{[/\\]+}}a.obj"

// RUN: %clang_cl /Fofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FoDIRNAMEEXT %s
// FoDIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /Fo.. -### -- %s 2>&1 | FileCheck -check-prefix=FoCRAZY %s
// FoCRAZY:  "-o" "..obj"

// RUN: not %clang_cl /Foa.obj -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEERROR %s
// CHECK-MULTIPLESOURCEERROR: error: cannot specify '/Foa.obj' when compiling multiple source files

// RUN: %clang_cl /Fomydir/ -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEOK %s
// CHECK-MULTIPLESOURCEOK: "-o" "mydir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /Fo -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEOK2 %s
// CHECK-MULTIPLESOURCEOK2: "-o" "cl-outputs.obj"
// CHECK-MULTIPLESOURCEOK2: "-o" "cl-outputs.obj"

// RUN: %clang_cl /c /oa -### -- %s 2>&1 | FileCheck -check-prefix=oNAME1 %s
// oNAME1:  "-o" "a.obj"

// RUN: %clang_cl /c /o a -### -- %s 2>&1 | FileCheck -check-prefix=oNAME2 %s
// oNAME2:  "-o" "a.obj"

// RUN: %clang_cl /c /oa.ext /ob.ext -### -- %s 2>&1 | FileCheck -check-prefix=oNAMEEXT1 %s
// oNAMEEXT1:  "-o" "b.ext"

// RUN: %clang_cl /c /o a.ext /ob.ext -### -- %s 2>&1 | FileCheck -check-prefix=oNAMEEXT2 %s
// oNAMEEXT2:  "-o" "b.ext"

// RUN: %clang_cl /c /ofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=oDIR1 %s
// oDIR1:  "-o" "foo.dir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /c /o foo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=oDIR2 %s
// oDIR2:  "-o" "foo.dir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /c /ofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=oDIRNAME1 %s
// oDIRNAME1:  "-o" "foo.dir{{[/\\]+}}a.obj"

// RUN: %clang_cl /c /o foo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=oDIRNAME2 %s
// oDIRNAME2:  "-o" "foo.dir{{[/\\]+}}a.obj"

// RUN: %clang_cl /c /ofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=oDIRNAMEEXT1 %s
// oDIRNAMEEXT1:  "-o" "foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /c /o foo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=oDIRNAMEEXT2 %s
// oDIRNAMEEXT2:  "-o" "foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /c /o.. -### -- %s 2>&1 | FileCheck -check-prefix=oCRAZY1 %s
// oCRAZY1:  "-o" "..obj"

// RUN: %clang_cl /c /o .. -### -- %s 2>&1 | FileCheck -check-prefix=oCRAZY2 %s
// oCRAZY2:  "-o" "..obj"

// RUN: not %clang_cl /c %s -### /o 2>&1 | FileCheck -check-prefix=oMISSINGARG %s
// oMISSINGARG: error: argument to '/o' is missing (expected 1 value)

// RUN: %clang_cl /c /omydir/ -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-oMULTIPLESOURCEOK1 %s
// CHECK-oMULTIPLESOURCEOK1: "-o" "mydir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /c /o mydir/ -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-oMULTIPLESOURCEOK2 %s
// CHECK-oMULTIPLESOURCEOK2: "-o" "mydir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl -emit-ast /otest.ast -###  -- %s  2>&1 | FileCheck -check-prefix=oASTNAME %s
// oASTNAME:  "-o" "test.ast"

// RUN: %clang_cl --analyze /otest.plist -###  -- %s  2>&1 | FileCheck -check-prefix=oPLIST %s
// oPLIST:  "-o" "test.plist"

// RUN: %clang_cl /c /obar /Fofoo -### -- %s 2>&1 | FileCheck -check-prefix=FooRACE1 %s
// FooRACE1: "-o" "foo.obj"

// RUN: %clang_cl /c /Fofoo /obar -### -- %s 2>&1 | FileCheck -check-prefix=FooRACE2 %s
// FooRACE2: "-o" "bar.obj"


// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTEXE %s
// DEFAULTEXE: cl-outputs.exe

// RUN: %clang_cl /LD -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTDLL %s
// RUN: %clang_cl /LDd -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTDLL %s
// DEFAULTDLL: "-out:cl-outputs.dll"
// DEFAULTDLL: "-implib:cl-outputs.lib"

// RUN: %clang_cl /Fefoo -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXT %s
// RUN: %clang_cl /Fe:foo -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXT %s
// FeNOEXT: "-out:foo.exe"

// RUN: %clang_cl /Fe -### -- %s 2>&1 | FileCheck -check-prefix=FeEMPTY %s
// FeEMPTY-NOT: argument to '/Fe' is missing
// FeEMPTY: "-out:cl-outputs.exe"

// RUN: %clang_cl /Fefoo /LD -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXTDLL %s
// RUN: %clang_cl /Fefoo /LDd -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXTDLL %s
// FeNOEXTDLL: "-out:foo.dll"
// FeNOEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// RUN: %clang_cl /Fe:foo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// RUN: %clang_cl /Fe: foo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// FeEXT: "-out:foo.ext"

// RUN: %clang_cl /LD /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXTDLL %s
// RUN: %clang_cl /LDd /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXTDLL %s
// FeEXTDLL: "-out:foo.ext"
// FeEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIR %s
// FeDIR: "-out:foo.dir{{[/\\]+}}cl-outputs.exe"

// RUN: %clang_cl /LD /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRDLL %s
// FeDIRDLL: "-out:foo.dir{{[/\\]+}}cl-outputs.dll"
// FeDIRDLL: "-implib:foo.dir{{[/\\]+}}cl-outputs.lib"

// RUN: %clang_cl /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAME %s
// FeDIRNAME: "-out:foo.dir{{[/\\]+}}a.exe"

// RUN: %clang_cl /LD /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEDLL %s
// FeDIRNAMEDLL: "-out:foo.dir{{[/\\]+}}a.dll"
// FeDIRNAMEDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: %clang_cl /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXT %s
// FeDIRNAMEEXT: "-out:foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /LD /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXTDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXTDLL %s
// FeDIRNAMEEXTDLL: "-out:foo.dir{{[/\\]+}}a.ext"
// FeDIRNAMEEXTDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: %clang_cl /Fefoo /Febar -### -- %s 2>&1 | FileCheck -check-prefix=FeOVERRIDE %s
// FeOVERRIDE: "-out:bar.exe"


// RUN: %clang_cl /obar /Fefoo -### -- %s 2>&1 | FileCheck -check-prefix=FeoRACE1 %s
// FeoRACE1: "-out:foo.exe"

// RUN: %clang_cl /Fefoo /obar -### -- %s 2>&1 | FileCheck -check-prefix=FeoRACE2 %s
// FeoRACE2: "-out:bar.exe"


// RUN: %clang_cl /ofoo -### -- %s 2>&1 | FileCheck -check-prefix=FeoNOEXT1 %s
// FeoNOEXT1: "-out:foo.exe"

// RUN: %clang_cl /o foo -### -- %s 2>&1 | FileCheck -check-prefix=FeoNOEXT2 %s
// FeoNOEXT2: "-out:foo.exe"

// RUN: %clang_cl /o foo /LD -### -- %s 2>&1 | FileCheck -check-prefix=FeoNOEXTDLL %s
// RUN: %clang_cl /ofoo /LDd -### -- %s 2>&1 | FileCheck -check-prefix=FeoNOEXTDLL %s
// FeoNOEXTDLL: "-out:foo.dll"
// FeoNOEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /ofoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoEXT1 %s
// FeoEXT1: "-out:foo.ext"

// RUN: %clang_cl /o foo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoEXT2 %s
// FeoEXT2: "-out:foo.ext"

// RUN: %clang_cl /LD /o foo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoEXTDLL %s
// RUN: %clang_cl /LDd /ofoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoEXTDLL %s
// FeoEXTDLL: "-out:foo.ext"
// FeoEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /ofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIR1 %s
// FeoDIR1: "-out:foo.dir{{[/\\]+}}cl-outputs.exe"

// RUN: %clang_cl /o foo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIR2 %s
// FeoDIR2: "-out:foo.dir{{[/\\]+}}cl-outputs.exe"

// RUN: %clang_cl /LD /o foo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRDLL %s
// RUN: %clang_cl /LDd /ofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRDLL %s
// FeoDIRDLL: "-out:foo.dir{{[/\\]+}}cl-outputs.dll"
// FeoDIRDLL: "-implib:foo.dir{{[/\\]+}}cl-outputs.lib"

// RUN: %clang_cl /ofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAME1 %s
// FeoDIRNAME1: "-out:foo.dir{{[/\\]+}}a.exe"

// RUN: %clang_cl /o foo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAME2 %s
// FeoDIRNAME2: "-out:foo.dir{{[/\\]+}}a.exe"

// RUN: %clang_cl /LD /o foo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEDLL %s
// RUN: %clang_cl /LDd /ofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEDLL %s
// FeoDIRNAMEDLL: "-out:foo.dir{{[/\\]+}}a.dll"
// FeoDIRNAMEDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: %clang_cl /ofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEEXT1 %s
// FeoDIRNAMEEXT1: "-out:foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /o foo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEEXT2 %s
// FeoDIRNAMEEXT2: "-out:foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /LD /o foo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEEXTDLL %s
// RUN: %clang_cl /LDd /ofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeoDIRNAMEEXTDLL %s
// FeoDIRNAMEEXTDLL: "-out:foo.dir{{[/\\]+}}a.ext"
// FeoDIRNAMEEXTDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: not %clang_cl -### /o 2>&1 | FileCheck -check-prefix=FeoMISSINGARG %s
// FeoMISSINGARG: error: argument to '/o' is missing (expected 1 value)

// RUN: %clang_cl /ofoo /o bar -### -- %s 2>&1 | FileCheck -check-prefix=FeoOVERRIDE %s
// FeoOVERRIDE: "-out:bar.exe"


// RUN: %clang_cl /FA -### -- %s 2>&1 | FileCheck -check-prefix=FA %s
// FA: "-o" "cl-outputs.asm"
// RUN: %clang_cl /FA /Fa -### -- %s 2>&1 | FileCheck -check-prefix=FaEMPTY %s
// FaEMPTY: "-o" "cl-outputs.asm"
// RUN: %clang_cl /FA /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FaNAME %s
// RUN: %clang_cl /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FaNAME %s
// FaNAME:  "-o" "foo.asm"
// RUN: %clang_cl /FA /Faa.ext /Fab.ext -### -- %s 2>&1 | FileCheck -check-prefix=FaNAMEEXT %s
// FaNAMEEXT:  "-o" "b.ext"
// RUN: %clang_cl /FA /Fafoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FaDIR %s
// FaDIR:  "-o" "foo.dir{{[/\\]+}}cl-outputs.asm"
// RUN: %clang_cl /FA /Fafoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FaDIRNAME %s
// FaDIRNAME:  "-o" "foo.dir{{[/\\]+}}a.asm"
// RUN: %clang_cl /FA /Fafoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FaDIRNAMEEXT %s
// FaDIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"
// RUN: not %clang_cl /Faa.asm -### -- %s %s 2>&1 | FileCheck -check-prefix=FaMULTIPLESOURCE %s
// FaMULTIPLESOURCE: error: cannot specify '/Faa.asm' when compiling multiple source files
// RUN: %clang_cl /Fa -### -- %s %s 2>&1 | FileCheck -check-prefix=FaMULTIPLESOURCEOK %s
// FaMULTIPLESOURCEOK: "-o" "cl-outputs.asm"
// FaMULTIPLESOURCEOK: "-o" "cl-outputs.asm"

// Copy of the same tests above, but with /FAcsu
// RUN: %clang_cl /FAcsu -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU %s
// FA_CSU: "-o" "cl-outputs.asm"
// RUN: %clang_cl /FAcsu /Fa -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaEMPTY %s
// FA_CSU_FaEMPTY: "-o" "cl-outputs.asm"
// RUN: %clang_cl /FAcsu /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaNAME %s
// RUN: %clang_cl /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaNAME %s
// FA_CSU_FaNAME:  "-o" "foo.asm"
// RUN: %clang_cl /FAcsu /Faa.ext /Fab.ext -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaNAMEEXT %s
// FA_CSU_FaNAMEEXT:  "-o" "b.ext"
// RUN: %clang_cl /FAcsu /Fafoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaDIR %s
// FA_CSU_FaDIR:  "-o" "foo.dir{{[/\\]+}}cl-outputs.asm"
// RUN: %clang_cl /FAcsu /Fafoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaDIRNAME %s
// FA_CSU_FaDIRNAME:  "-o" "foo.dir{{[/\\]+}}a.asm"
// RUN: %clang_cl /FAcsu /Fafoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaDIRNAMEEXT %s
// FA_CSU_FaDIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"
// RUN: not %clang_cl /Faa.asm -### -- %s %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaMULTIPLESOURCE %s
// FA_CSU_FaMULTIPLESOURCE: error: cannot specify '/Faa.asm' when compiling multiple source files
// RUN: %clang_cl /Fa -### -- %s %s 2>&1 | FileCheck -check-prefix=FA_CSU_FaMULTIPLESOURCEOK %s
// FA_CSU_FaMULTIPLESOURCEOK: "-o" "cl-outputs.asm"
// FA_CSU_FaMULTIPLESOURCEOK: "-o" "cl-outputs.asm"

// RUN: %clang_cl /P -### -- %s 2>&1 | FileCheck -check-prefix=P %s
// P: "-E"
// P: "-o" "cl-outputs.i"

// RUN: %clang_cl /P /Fifoo -### -- %s 2>&1 | FileCheck -check-prefix=Fi1 %s
// RUN: %clang_cl /P /Fi:foo -### -- %s 2>&1 | FileCheck -check-prefix=Fi1 %s
// RUN: %clang_cl /P /Fi: foo -### -- %s 2>&1 | FileCheck -check-prefix=Fi1 %s
// Fi1: "-E"
// Fi1: "-o" "foo.i"

// RUN: %clang_cl /P /Fifoo.x -### -- %s 2>&1 | FileCheck -check-prefix=Fi2 %s
// Fi2: "-E"
// Fi2: "-o" "foo.x"

// To match MSVC behavior /o should be ignored for /P output.

// RUN: %clang_cl /P /ofoo -### -- %s 2>&1 | FileCheck -check-prefix=Fio1 %s
// Fio1: "-E"
// Fio1: "-o" "cl-outputs.i"

// RUN: %clang_cl /P /o foo.x -### -- %s 2>&1 | FileCheck -check-prefix=Fio2 %s
// Fio2: "-E"
// Fio2: "-o" "cl-outputs.i"

// RUN: %clang_cl /P /obar.x /Fifoo.x -### -- %s 2>&1 | FileCheck -check-prefix=FioRACE1 %s
// FioRACE1: "-E"
// FioRACE1: "-o" "foo.x"

// RUN: %clang_cl /P /Fifoo.x /obar.x -### -- %s 2>&1 | FileCheck -check-prefix=FioRACE2 %s
// FioRACE2: "-E"
// FioRACE2: "-o" "foo.x"

// RUN: %clang_cl -target x86_64-windows /Z7 /Foa.obj -### -- %s 2>&1 | FileCheck -check-prefix=ABSOLUTE_OBJPATH %s
// ABSOLUTE_OBJPATH: "-object-file-name={{.*}}a.obj"

// RUN: %clang_cl -target x86_64-windows -fdebug-compilation-dir=. /Z7 /Foa.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH1 %s
// RELATIVE_OBJPATH1: "-object-file-name=a.obj"

// RUN: %clang_cl -target x86_64-windows -fdebug-compilation-dir=. /Z7 /Fo:a.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH1_COLON %s
// RUN: %clang_cl -target x86_64-windows -fdebug-compilation-dir=. /Z7 /Fo: a.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH1_COLON %s
// RELATIVE_OBJPATH1_COLON: "-object-file-name=a.obj"

// RUN: %clang_cl -target x86_64-windows -fdebug-compilation-dir=. /Z7 /Fofoo/a.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH2 %s
// RELATIVE_OBJPATH2: "-object-file-name=foo\\a.obj"
