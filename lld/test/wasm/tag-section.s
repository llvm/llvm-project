# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling -o %t_tags.o %p/Inputs/tags.s

# Static code, with tags defined in tags.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %p/Inputs/tag-section1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %p/Inputs/tag-section2.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %s -o %t.o
# RUN: wasm-ld --export=my_global -o %t.wasm %t.o %t1.o %t2.o %t_tags.o
# RUN: wasm-ld --export-all -o %t-export-all.wasm %t.o %t1.o %t2.o %t_tags.o
# RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=NOPIC
# RUN: obj2yaml %t-export-all.wasm | FileCheck %s --check-prefix=NOPIC-EXPORT-ALL

# PIC code with tags imported
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %p/Inputs/tag-section1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %p/Inputs/tag-section2.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+exception-handling %s -o %t.o
# RUN: wasm-ld --import-undefined --unresolved-symbols=import-dynamic -pie -o %t_pic.wasm %t.o %t1.o %t2.o
# RUN: obj2yaml %t_pic.wasm | FileCheck %s --check-prefix=PIC

.functype foo (i32) -> ()
.functype bar (i32) -> ()
.globl _start

_start:
  .functype _start () -> ()
  i32.const 0
  call foo
  i32.const 0
  call bar
  end_function

.globl my_global
.section .data.my_global,"",@
.p2align 2
my_global:
  .int32 42
  .size my_global, 4

# NOPIC:      Sections:
# NOPIC-NEXT:   - Type:            TYPE
# NOPIC-NEXT:     Signatures:
# NOPIC-NEXT:       - Index:           0
# NOPIC-NEXT:         ParamTypes:      []
# NOPIC-NEXT:         ReturnTypes:     []
# NOPIC-NEXT:       - Index:           1
# NOPIC-NEXT:         ParamTypes:
# NOPIC-NEXT:           - I32
# NOPIC-NEXT:         ReturnTypes:     []

# NOPIC:        - Type:            TAG
# NOPIC-NEXT:     TagTypes:        [ 1 ]

# Global section has to come after tag section
# NOPIC:        - Type:            GLOBAL

# NOPIC-EXPORT-ALL:   - Type:            EXPORT
# NOPIC-EXPORT-ALL-NEXT Exports:
# NOPIC-EXPORT-ALL:       - Name:            __cpp_exception
# NOPIC-EXPORT-ALL:         Kind:            TAG
# NOPIC-EXPORT-ALL:         Index:           0

# In PIC mode, we leave the tags as undefined and they should be imported
# PIC:        Sections:
# PIC:         - Type:            TYPE
# PIC-NEXT:      Signatures:
# PIC-NEXT:        - Index:           0
# PIC-NEXT:          ParamTypes:
# PIC-NEXT:            - I32
# PIC-NEXT:          ReturnTypes:     []
# PIC-NEXT:        - Index:           1
# PIC-NEXT:          ParamTypes:      []
# PIC-NEXT:          ReturnTypes:     []

# PIC:         - Type:            IMPORT
# PIC-NEXT:      Imports:
# PIC:             - Module:          env
# PIC:               Field:           __cpp_exception
# PIC-NEXT:          Kind:            TAG
# PIC-NEXT:          SigIndex:        0

# In PIC mode, tags should NOT be defined in the module; they are imported.
# PIC-NOT:     - Type:            TAG
