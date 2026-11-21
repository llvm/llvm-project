# REQUIRES: x86

## /dependency-file: lists every file the link reads: libraries found through
## /defaultlib: directives and /libpath:, the members of a thin archive that
## the link extracts (and not the others), all members of a thin archive given
## with /wholearchive:, and files named by options.

# RUN: rm -rf %t.dir && split-file %s %t.dir && cd %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows main.s -o main.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows used.s -o used.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows unused.s -o unused.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows whole.s -o whole.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows regular.s -o regular.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows dl.s -o lib/dl.obj
# RUN: llvm-ar rcsT thin.lib used.obj unused.obj
# RUN: llvm-ar rcsT whole.lib whole.obj
# RUN: llvm-ar rcs regular.lib regular.obj
# RUN: llvm-ar rcs lib/dl.lib lib/dl.obj

# RUN: lld-link /out:out.dll /dll /entry:main /debug /pdb:out.pdb \
# RUN:   /libpath:lib main.obj thin.lib /wholearchive:whole.lib regular.lib \
# RUN:   /def:exports.def /natvis:types.natvis /order:@order.txt \
# RUN:   /dependency-file:out.d
# RUN: FileCheck --match-full-lines %s < out.d

# CHECK:      out.dll: \
# CHECK-NEXT:   main.obj \
# CHECK-NEXT:   thin.lib \
# CHECK-NEXT:   whole.lib \
# CHECK-NEXT:   whole.obj \
# CHECK-NEXT:   regular.lib \
# CHECK-NEXT:   lib{{/|\\}}dl.lib \
# CHECK-NEXT:   used.obj \
# CHECK-NEXT:   exports.def \
# CHECK-NEXT:   order.txt \
# CHECK-NEXT:   types.natvis
# CHECK-EMPTY:
# CHECK-NEXT: main.obj:
# CHECK-NOT:  unused.obj

## Files that other options name. The manifest resource that /manifest:embed
## makes in memory is not a file.
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows used.s -o used2.obj
# RUN: lld-link /out:out2.exe /entry:used /subsystem:console used2.obj \
# RUN:   regular.obj /stub:%p/Inputs/stub64mz /sectionlayout:layout.txt \
# RUN:   /call-graph-ordering-file:cg.txt /vfsoverlay:overlay.yaml \
# RUN:   /lto-sample-profile:sample.prof /debug /pdb:out2.pdb \
# RUN:   /pdbstream:extra=stream.txt /manifest:embed /dependency-file:out2.d
# RUN: FileCheck --check-prefix=OPTIONS --match-full-lines %s < out2.d

# OPTIONS:      out2.exe: \
# OPTIONS-NEXT:   overlay.yaml \
# OPTIONS-NEXT:   layout.txt \
# OPTIONS-NEXT:   sample.prof \
# OPTIONS-NEXT:   used2.obj \
# OPTIONS-NEXT:   regular.obj \
# OPTIONS-NEXT:   {{.*}}Inputs{{/|\\}}stub64mz \
# OPTIONS-NEXT:   cg.txt \
# OPTIONS-NEXT:   stream.txt
# OPTIONS-EMPTY:
# OPTIONS-NOT:  manifest.res

#--- main.s
.globl main
main:
  call used
  ret

.section .drectve,"yn"
.ascii " /defaultlib:dl.lib"

#--- used.s
.globl used
used:
  call regular
  ret

#--- unused.s
.globl unused
unused:
  ret

#--- whole.s
.globl whole
whole:
  ret

#--- regular.s
.globl regular
regular:
  ret

#--- dl.s
.globl dl
dl:
  ret

#--- lib/.keep

#--- exports.def
EXPORTS
  main

#--- order.txt
main

#--- layout.txt
.text

#--- cg.txt
used regular 10

#--- overlay.yaml
{ 'version': 0, 'roots': [] }

#--- sample.prof

#--- stream.txt
extra

#--- types.natvis
<?xml version="1.0" encoding="utf-8"?>
<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010">
</AutoVisualizer>
