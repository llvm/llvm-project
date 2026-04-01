" Vim syntax file
" Language:   aiir
" Maintainer: The AIIR team, http://github.com/tensorflow/aiir/
" Version:      $Revision$
" Some parts adapted from the LLVM vim syntax file.

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
"
syn keyword aiirType index f16 f32 f64 bf16
" Signless integer types.
syn match aiirType /\<i\d\+\>/
" Unsigned integer types.
syn match aiirType /\<ui\d\+\>/
" Signed integer types.
syn match aiirType /\<si\d\+\>/

" Elemental types inside memref, tensor, or vector types.
syn match aiirType /x\s*\zs\(bf16|f16\|f32\|f64\|i\d\+\|ui\d\+\|si\d\+\)/

" Shaped types.
syn match aiirType /\<memref\ze\s*<.*>/
syn match aiirType /\<tensor\ze\s*<.*>/
syn match aiirType /\<vector\ze\s*<.*>/

" vector types inside memref or tensor.
syn match aiirType /x\s*\zsvector/

" Operations.
" TODO: this list is not exhaustive.
syn keyword aiirOps alloc alloca addf addi and call call_indirect cmpf cmpi
syn keyword aiirOps constant dealloc divf dma_start dma_wait dim exp
syn keyword aiirOps getTensor index_cast load log memref_cast
syn keyword aiirOps memref_shape_cast mulf muli negf powf prefetch rsqrt sitofp
syn keyword aiirOps splat store select sqrt subf subi subview tanh
syn keyword aiirOps view

" Math ops.
syn match aiirOps /\<math\.erf\>/
syn match aiirOps /\<math\.erfc\>/

" Affine ops.
syn match aiirOps /\<affine\.apply\>/
syn match aiirOps /\<affine\.dma_start\>/
syn match aiirOps /\<affine\.dma_wait\>/
syn match aiirOps /\<affine\.for\>/
syn match aiirOps /\<affine\.if\>/
syn match aiirOps /\<affine\.load\>/
syn match aiirOps /\<affine\.parallel\>/
syn match aiirOps /\<affine\.prefetch\>/
syn match aiirOps /\<affine\.store\>/
syn match aiirOps /\<scf\.execute_region\>/
syn match aiirOps /\<scf\.for\>/
syn match aiirOps /\<scf\.if\>/
syn match aiirOps /\<scf\.yield\>/

" TODO: dialect name prefixed ops (llvm or std).

" Keywords.
syn keyword aiirKeyword
      \ affine_map
      \ affine_set
      \ dense
      \ else
      \ func
      \ module
      \ return
      \ step
      \ to

" Misc syntax.

syn match   aiirNumber /-\?\<\d\+\>/
" Match numbers even in shaped types.
syn match   aiirNumber /-\?\<\d\+\ze\s*x/
syn match   aiirNumber /x\s*\zs-\?\d\+\ze\s*x/

syn match   aiirFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   aiirFloat  /\<0x\x\+\>/
syn keyword aiirBoolean true false
" Spell checking is enabled only in comments by default.
syn match   aiirComment /\/\/.*$/ contains=@Spell
syn region  aiirString start=/"/ skip=/\\"/ end=/"/
syn match   aiirLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
" Prefixed identifiers usually used for ssa values and symbols.
syn match   aiirIdentifier /[%@][a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   aiirIdentifier /[%@]\d\+\>/
" Prefixed identifiers usually used for blocks.
syn match   aiirBlockIdentifier /\^[a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   aiirBlockIdentifier /\^\d\+\>/
" Prefixed identifiers usually used for types.
syn match   aiirTypeIdentifier /![a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   aiirTypeIdentifier /!\d\+\>/
" Prefixed identifiers usually used for attribute aliases and result numbers.
syn match   aiirAttrIdentifier /#[a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   aiirAttrIdentifier /#\d\+\>/

" Syntax-highlight lit test commands and bug numbers.
syn match  aiirSpecialComment /\/\/\s*RUN:.*$/
syn match  aiirSpecialComment /\/\/\s*CHECK:.*$/
syn match  aiirSpecialComment "\v\/\/\s*CHECK-(NEXT|NOT|DAG|SAME|LABEL):.*$"
syn match  aiirSpecialComment /\/\/\s*expected-error.*$/
syn match  aiirSpecialComment /\/\/\s*expected-remark.*$/
syn match  aiirSpecialComment /;\s*XFAIL:.*$/
syn match  aiirSpecialComment /\/\/\s*PR\d*\s*$/
syn match  aiirSpecialComment /\/\/\s*REQUIRES:.*$/

if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink aiirType Type
  HiLink aiirOps Statement
  HiLink aiirNumber Number
  HiLink aiirComment Comment
  HiLink aiirString String
  HiLink aiirLabel Label
  HiLink aiirKeyword Keyword
  HiLink aiirBoolean Boolean
  HiLink aiirFloat Float
  HiLink aiirConstant Constant
  HiLink aiirSpecialComment SpecialComment
  HiLink aiirIdentifier Identifier
  HiLink aiirBlockIdentifier Label
  HiLink aiirTypeIdentifier Type
  HiLink aiirAttrIdentifier PreProc

  delcommand HiLink
endif

let b:current_syntax = "aiir"
