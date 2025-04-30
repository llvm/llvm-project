/* DWARF2 exception handling and frame unwind runtime interface routines.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifdef _LIBC
#include <stdlib.h>
#include <string.h>
#include <error.h>
#include <libintl.h>
#include <dwarf2.h>
#include <stdio.h>
#include <unwind.h>
#include <unwind-pe.h>
#include <unwind-dw2-fde.h>
#else
#include "tconfig.h"
#include "tsystem.h"
#include "dwarf2.h"
#include "unwind.h"
#include "unwind-pe.h"
#include "unwind-dw2-fde.h"
#include "gthr.h"
#endif



#ifndef STACK_GROWS_DOWNWARD
#define STACK_GROWS_DOWNWARD 0
#else
#undef STACK_GROWS_DOWNWARD
#define STACK_GROWS_DOWNWARD 1
#endif

/* A target can override (perhaps for backward compatibility) how
   many dwarf2 columns are unwound.  */
#ifndef DWARF_FRAME_REGISTERS
#define DWARF_FRAME_REGISTERS FIRST_PSEUDO_REGISTER
#endif

/* Dwarf frame registers used for pre gcc 3.0 compiled glibc.  */
#ifndef PRE_GCC3_DWARF_FRAME_REGISTERS
#define PRE_GCC3_DWARF_FRAME_REGISTERS DWARF_FRAME_REGISTERS
#endif

/* This is the register and unwind state for a particular frame.  This
   provides the information necessary to unwind up past a frame and return
   to its caller.  */
struct _Unwind_Context
{
  void *reg[DWARF_FRAME_REGISTERS+1];
  void *cfa;
  void *ra;
  void *lsda;
  struct dwarf_eh_bases bases;
  _Unwind_Word args_size;
};

#ifndef _LIBC
/* Byte size of every register managed by these routines.  */
static unsigned char dwarf_reg_size_table[DWARF_FRAME_REGISTERS];
#endif


/* The result of interpreting the frame unwind info for a frame.
   This is all symbolic at this point, as none of the values can
   be resolved until the target pc is located.  */
typedef struct
{
  /* Each register save state can be described in terms of a CFA slot,
     another register, or a location expression.  */
  struct frame_state_reg_info
  {
    struct {
      union {
	_Unwind_Word reg;
	_Unwind_Sword offset;
	const unsigned char *exp;
      } loc;
      enum {
	REG_UNSAVED,
	REG_SAVED_OFFSET,
	REG_SAVED_REG,
	REG_SAVED_EXP,
      } how;
    } reg[DWARF_FRAME_REGISTERS+1];

    /* Used to implement DW_CFA_remember_state.  */
    struct frame_state_reg_info *prev;
  } regs;

  /* The CFA can be described in terms of a reg+offset or a
     location expression.  */
  _Unwind_Sword cfa_offset;
  _Unwind_Word cfa_reg;
  const unsigned char *cfa_exp;
  enum {
    CFA_UNSET,
    CFA_REG_OFFSET,
    CFA_EXP,
  } cfa_how;

  /* The PC described by the current frame state.  */
  void *pc;

  /* The information we care about from the CIE/FDE.  */
  _Unwind_Personality_Fn personality;
  _Unwind_Sword data_align;
  _Unwind_Word code_align;
  unsigned char retaddr_column;
  unsigned char fde_encoding;
  unsigned char lsda_encoding;
  unsigned char saw_z;
  void *eh_ptr;
} _Unwind_FrameState;

/* Read unaligned data from the instruction buffer.  */

union unaligned
{
  void *p;
  unsigned u2 __attribute__ ((mode (HI)));
  unsigned u4 __attribute__ ((mode (SI)));
  unsigned u8 __attribute__ ((mode (DI)));
  signed s2 __attribute__ ((mode (HI)));
  signed s4 __attribute__ ((mode (SI)));
  signed s8 __attribute__ ((mode (DI)));
} __attribute__ ((packed));

static inline void *
read_pointer (const void *p) { const union unaligned *up = p; return up->p; }

static inline int
read_1u (const void *p) { return *(const unsigned char *) p; }

static inline int
read_1s (const void *p) { return *(const signed char *) p; }

static inline int
read_2u (const void *p) { const union unaligned *up = p; return up->u2; }

static inline int
read_2s (const void *p) { const union unaligned *up = p; return up->s2; }

static inline unsigned int
read_4u (const void *p) { const union unaligned *up = p; return up->u4; }

static inline int
read_4s (const void *p) { const union unaligned *up = p; return up->s4; }

static inline unsigned long
read_8u (const void *p) { const union unaligned *up = p; return up->u8; }

static inline unsigned long
read_8s (const void *p) { const union unaligned *up = p; return up->s8; }

/* Get the value of register REG as saved in CONTEXT.  */

inline _Unwind_Word
_Unwind_GetGR (struct _Unwind_Context *context, int index)
{
  /* This will segfault if the register hasn't been saved.  */
  return * (_Unwind_Word *) context->reg[index];
}

/* Overwrite the saved value for register REG in CONTEXT with VAL.  */

inline void
_Unwind_SetGR (struct _Unwind_Context *context, int index, _Unwind_Word val)
{
  * (_Unwind_Word *) context->reg[index] = val;
}

/* Retrieve the return address for CONTEXT.  */

inline _Unwind_Ptr
_Unwind_GetIP (struct _Unwind_Context *context)
{
  return (_Unwind_Ptr) context->ra;
}

/* Overwrite the return address for CONTEXT with VAL.  */

inline void
_Unwind_SetIP (struct _Unwind_Context *context, _Unwind_Ptr val)
{
  context->ra = (void *) val;
}

void *
_Unwind_GetLanguageSpecificData (struct _Unwind_Context *context)
{
  return context->lsda;
}

_Unwind_Ptr
_Unwind_GetRegionStart (struct _Unwind_Context *context)
{
  return (_Unwind_Ptr) context->bases.func;
}

void *
_Unwind_FindEnclosingFunction (void *pc)
{
  struct dwarf_eh_bases bases;
  struct dwarf_fde *fde = _Unwind_Find_FDE (pc-1, &bases);
  if (fde)
    return bases.func;
  else
    return NULL;
}

#ifndef __ia64__
_Unwind_Ptr
_Unwind_GetDataRelBase (struct _Unwind_Context *context)
{
  return (_Unwind_Ptr) context->bases.dbase;
}

_Unwind_Ptr
_Unwind_GetTextRelBase (struct _Unwind_Context *context)
{
  return (_Unwind_Ptr) context->bases.tbase;
}
#endif

/* Extract any interesting information from the CIE for the translation
   unit F belongs to.  Return a pointer to the byte after the augmentation,
   or NULL if we encountered an undecipherable augmentation.  */

static const unsigned char *
extract_cie_info (struct dwarf_cie *cie, struct _Unwind_Context *context,
		  _Unwind_FrameState *fs)
{
  const unsigned char *aug = cie->augmentation;
  const unsigned char *p = aug + strlen ((const char *) aug) + 1;
  const unsigned char *ret = NULL;
  _Unwind_Word utmp;

  /* g++ v2 "eh" has pointer immediately following augmentation string,
     so it must be handled first.  */
  if (aug[0] == 'e' && aug[1] == 'h')
    {
      fs->eh_ptr = read_pointer (p);
      p += sizeof (void *);
      aug += 2;
    }

  /* Immediately following the augmentation are the code and
     data alignment and return address column.  */
  p = read_uleb128 (p, &fs->code_align);
  p = read_sleb128 (p, &fs->data_align);
  fs->retaddr_column = *p++;
  fs->lsda_encoding = DW_EH_PE_omit;

  /* If the augmentation starts with 'z', then a uleb128 immediately
     follows containing the length of the augmentation field following
     the size.  */
  if (*aug == 'z')
    {
      p = read_uleb128 (p, &utmp);
      ret = p + utmp;

      fs->saw_z = 1;
      ++aug;
    }

  /* Iterate over recognized augmentation subsequences.  */
  while (*aug != '\0')
    {
      /* "L" indicates a byte showing how the LSDA pointer is encoded.  */
      if (aug[0] == 'L')
	{
	  fs->lsda_encoding = *p++;
	  aug += 1;
	}

      /* "R" indicates a byte indicating how FDE addresses are encoded.  */
      else if (aug[0] == 'R')
	{
	  fs->fde_encoding = *p++;
	  aug += 1;
	}

      /* "P" indicates a personality routine in the CIE augmentation.  */
      else if (aug[0] == 'P')
	{
	  _Unwind_Ptr personality;
	  p = read_encoded_value (context, *p, p + 1, &personality);
	  fs->personality = (_Unwind_Personality_Fn) personality;
	  aug += 1;
	}

      /* Otherwise we have an unknown augmentation string.
	 Bail unless we saw a 'z' prefix.  */
      else
	return ret;
    }

  return ret ? ret : p;
}

#ifndef _LIBC
/* Decode a DW_OP stack program.  Return the top of stack.  Push INITIAL
   onto the stack to start.  */

static _Unwind_Word
execute_stack_op (const unsigned char *op_ptr, const unsigned char *op_end,
		  struct _Unwind_Context *context, _Unwind_Word initial)
{
  _Unwind_Word stack[64];	/* ??? Assume this is enough.  */
  int stack_elt;

  stack[0] = initial;
  stack_elt = 1;

  while (op_ptr < op_end)
    {
      enum dwarf_location_atom op = *op_ptr++;
      _Unwind_Word result, reg, utmp;
      _Unwind_Sword offset, stmp;

      switch (op)
	{
	case DW_OP_lit0:
	case DW_OP_lit1:
	case DW_OP_lit2:
	case DW_OP_lit3:
	case DW_OP_lit4:
	case DW_OP_lit5:
	case DW_OP_lit6:
	case DW_OP_lit7:
	case DW_OP_lit8:
	case DW_OP_lit9:
	case DW_OP_lit10:
	case DW_OP_lit11:
	case DW_OP_lit12:
	case DW_OP_lit13:
	case DW_OP_lit14:
	case DW_OP_lit15:
	case DW_OP_lit16:
	case DW_OP_lit17:
	case DW_OP_lit18:
	case DW_OP_lit19:
	case DW_OP_lit20:
	case DW_OP_lit21:
	case DW_OP_lit22:
	case DW_OP_lit23:
	case DW_OP_lit24:
	case DW_OP_lit25:
	case DW_OP_lit26:
	case DW_OP_lit27:
	case DW_OP_lit28:
	case DW_OP_lit29:
	case DW_OP_lit30:
	case DW_OP_lit31:
	  result = op - DW_OP_lit0;
	  break;

	case DW_OP_addr:
	  result = (_Unwind_Word) (_Unwind_Ptr) read_pointer (op_ptr);
	  op_ptr += sizeof (void *);
	  break;

	case DW_OP_const1u:
	  result = read_1u (op_ptr);
	  op_ptr += 1;
	  break;
	case DW_OP_const1s:
	  result = read_1s (op_ptr);
	  op_ptr += 1;
	  break;
	case DW_OP_const2u:
	  result = read_2u (op_ptr);
	  op_ptr += 2;
	  break;
	case DW_OP_const2s:
	  result = read_2s (op_ptr);
	  op_ptr += 2;
	  break;
	case DW_OP_const4u:
	  result = read_4u (op_ptr);
	  op_ptr += 4;
	  break;
	case DW_OP_const4s:
	  result = read_4s (op_ptr);
	  op_ptr += 4;
	  break;
	case DW_OP_const8u:
	  result = read_8u (op_ptr);
	  op_ptr += 8;
	  break;
	case DW_OP_const8s:
	  result = read_8s (op_ptr);
	  op_ptr += 8;
	  break;
	case DW_OP_constu:
	  op_ptr = read_uleb128 (op_ptr, &result);
	  break;
	case DW_OP_consts:
	  op_ptr = read_sleb128 (op_ptr, &stmp);
	  result = stmp;
	  break;

	case DW_OP_reg0:
	case DW_OP_reg1:
	case DW_OP_reg2:
	case DW_OP_reg3:
	case DW_OP_reg4:
	case DW_OP_reg5:
	case DW_OP_reg6:
	case DW_OP_reg7:
	case DW_OP_reg8:
	case DW_OP_reg9:
	case DW_OP_reg10:
	case DW_OP_reg11:
	case DW_OP_reg12:
	case DW_OP_reg13:
	case DW_OP_reg14:
	case DW_OP_reg15:
	case DW_OP_reg16:
	case DW_OP_reg17:
	case DW_OP_reg18:
	case DW_OP_reg19:
	case DW_OP_reg20:
	case DW_OP_reg21:
	case DW_OP_reg22:
	case DW_OP_reg23:
	case DW_OP_reg24:
	case DW_OP_reg25:
	case DW_OP_reg26:
	case DW_OP_reg27:
	case DW_OP_reg28:
	case DW_OP_reg29:
	case DW_OP_reg30:
	case DW_OP_reg31:
	  result = _Unwind_GetGR (context, op - DW_OP_reg0);
	  break;
	case DW_OP_regx:
	  op_ptr = read_uleb128 (op_ptr, &reg);
	  result = _Unwind_GetGR (context, reg);
	  break;

	case DW_OP_breg0:
	case DW_OP_breg1:
	case DW_OP_breg2:
	case DW_OP_breg3:
	case DW_OP_breg4:
	case DW_OP_breg5:
	case DW_OP_breg6:
	case DW_OP_breg7:
	case DW_OP_breg8:
	case DW_OP_breg9:
	case DW_OP_breg10:
	case DW_OP_breg11:
	case DW_OP_breg12:
	case DW_OP_breg13:
	case DW_OP_breg14:
	case DW_OP_breg15:
	case DW_OP_breg16:
	case DW_OP_breg17:
	case DW_OP_breg18:
	case DW_OP_breg19:
	case DW_OP_breg20:
	case DW_OP_breg21:
	case DW_OP_breg22:
	case DW_OP_breg23:
	case DW_OP_breg24:
	case DW_OP_breg25:
	case DW_OP_breg26:
	case DW_OP_breg27:
	case DW_OP_breg28:
	case DW_OP_breg29:
	case DW_OP_breg30:
	case DW_OP_breg31:
	  op_ptr = read_sleb128 (op_ptr, &offset);
	  result = _Unwind_GetGR (context, op - DW_OP_breg0) + offset;
	  break;
	case DW_OP_bregx:
	  op_ptr = read_uleb128 (op_ptr, &reg);
	  op_ptr = read_sleb128 (op_ptr, &offset);
	  result = _Unwind_GetGR (context, reg) + offset;
	  break;

	case DW_OP_dup:
	  if (stack_elt < 1)
	    abort ();
	  result = stack[stack_elt - 1];
	  break;

	case DW_OP_drop:
	  if (--stack_elt < 0)
	    abort ();
	  goto no_push;

	case DW_OP_pick:
	  offset = *op_ptr++;
	  if (offset >= stack_elt - 1)
	    abort ();
	  result = stack[stack_elt - 1 - offset];
	  break;

	case DW_OP_over:
	  if (stack_elt < 2)
	    abort ();
	  result = stack[stack_elt - 2];
	  break;

	case DW_OP_rot:
	  {
	    _Unwind_Word t1, t2, t3;

	    if (stack_elt < 3)
	      abort ();
	    t1 = stack[stack_elt - 1];
	    t2 = stack[stack_elt - 2];
	    t3 = stack[stack_elt - 3];
	    stack[stack_elt - 1] = t2;
	    stack[stack_elt - 2] = t3;
	    stack[stack_elt - 3] = t1;
	    goto no_push;
	  }

	case DW_OP_deref:
	case DW_OP_deref_size:
	case DW_OP_abs:
	case DW_OP_neg:
	case DW_OP_not:
	case DW_OP_plus_uconst:
	  /* Unary operations.  */
	  if (--stack_elt < 0)
	    abort ();
	  result = stack[stack_elt];

	  switch (op)
	    {
	    case DW_OP_deref:
	      {
		void *ptr = (void *) (_Unwind_Ptr) result;
		result = (_Unwind_Ptr) read_pointer (ptr);
	      }
	      break;

	    case DW_OP_deref_size:
	      {
		void *ptr = (void *) (_Unwind_Ptr) result;
		switch (*op_ptr++)
		  {
		  case 1:
		    result = read_1u (ptr);
		    break;
		  case 2:
		    result = read_2u (ptr);
		    break;
		  case 4:
		    result = read_4u (ptr);
		    break;
		  case 8:
		    result = read_8u (ptr);
		    break;
		  default:
		    abort ();
		  }
	      }
	      break;

	    case DW_OP_abs:
	      if ((_Unwind_Sword) result < 0)
		result = -result;
	      break;
	    case DW_OP_neg:
	      result = -result;
	      break;
	    case DW_OP_not:
	      result = ~result;
	      break;
	    case DW_OP_plus_uconst:
	      op_ptr = read_uleb128 (op_ptr, &utmp);
	      result += utmp;
	      break;

	    default:
	      abort ();
	    }
	  break;

	case DW_OP_and:
	case DW_OP_div:
	case DW_OP_minus:
	case DW_OP_mod:
	case DW_OP_mul:
	case DW_OP_or:
	case DW_OP_plus:
	case DW_OP_le:
	case DW_OP_ge:
	case DW_OP_eq:
	case DW_OP_lt:
	case DW_OP_gt:
	case DW_OP_ne:
	  {
	    /* Binary operations.  */
	    _Unwind_Word first, second;
	    if ((stack_elt -= 2) < 0)
	      abort ();
	    second = stack[stack_elt];
	    first = stack[stack_elt + 1];

	    switch (op)
	      {
	      case DW_OP_and:
		result = second & first;
		break;
	      case DW_OP_div:
		result = (_Unwind_Sword) second / (_Unwind_Sword) first;
		break;
	      case DW_OP_minus:
		result = second - first;
		break;
	      case DW_OP_mod:
		result = (_Unwind_Sword) second % (_Unwind_Sword) first;
		break;
	      case DW_OP_mul:
		result = second * first;
		break;
	      case DW_OP_or:
		result = second | first;
		break;
	      case DW_OP_plus:
		result = second + first;
		break;
	      case DW_OP_shl:
		result = second << first;
		break;
	      case DW_OP_shr:
		result = second >> first;
		break;
	      case DW_OP_shra:
		result = (_Unwind_Sword) second >> first;
		break;
	      case DW_OP_xor:
		result = second ^ first;
		break;
	      case DW_OP_le:
		result = (_Unwind_Sword) first <= (_Unwind_Sword) second;
		break;
	      case DW_OP_ge:
		result = (_Unwind_Sword) first >= (_Unwind_Sword) second;
		break;
	      case DW_OP_eq:
		result = (_Unwind_Sword) first == (_Unwind_Sword) second;
		break;
	      case DW_OP_lt:
		result = (_Unwind_Sword) first < (_Unwind_Sword) second;
		break;
	      case DW_OP_gt:
		result = (_Unwind_Sword) first > (_Unwind_Sword) second;
		break;
	      case DW_OP_ne:
		result = (_Unwind_Sword) first != (_Unwind_Sword) second;
		break;

	      default:
		abort ();
	      }
	  }
	  break;

	case DW_OP_skip:
	  offset = read_2s (op_ptr);
	  op_ptr += 2;
	  op_ptr += offset;
	  goto no_push;

	case DW_OP_bra:
	  if (--stack_elt < 0)
	    abort ();
	  offset = read_2s (op_ptr);
	  op_ptr += 2;
	  if (stack[stack_elt] != 0)
	    op_ptr += offset;
	  goto no_push;

	case DW_OP_nop:
	  goto no_push;

	default:
	  abort ();
	}

      /* Most things push a result value.  */
      if ((size_t) stack_elt >= sizeof (stack) / sizeof (*stack))
	abort ();
      stack[stack_elt++] = result;
    no_push:;
    }

  /* We were executing this program to get a value.  It should be
     at top of stack.  */
  if (--stack_elt < 0)
    abort ();
  return stack[stack_elt];
}
#endif

/* Decode DWARF 2 call frame information. Takes pointers the
   instruction sequence to decode, current register information and
   CIE info, and the PC range to evaluate.  */

static void
execute_cfa_program (const unsigned char *insn_ptr,
		     const unsigned char *insn_end,
		     struct _Unwind_Context *context,
		     _Unwind_FrameState *fs)
{
  struct frame_state_reg_info *unused_rs = NULL;

  /* Don't allow remember/restore between CIE and FDE programs.  */
  fs->regs.prev = NULL;

  /* The comparison with the return address uses < rather than <= because
     we are only interested in the effects of code before the call; for a
     noreturn function, the return address may point to unrelated code with
     a different stack configuration that we are not interested in.  We
     assume that the call itself is unwind info-neutral; if not, or if
     there are delay instructions that adjust the stack, these must be
     reflected at the point immediately before the call insn.  */
  while (insn_ptr < insn_end && fs->pc < context->ra)
    {
      unsigned char insn = *insn_ptr++;
      _Unwind_Word reg, utmp;
      _Unwind_Sword offset, stmp;

      if ((insn & 0xc0) == DW_CFA_advance_loc)
	fs->pc += (insn & 0x3f) * fs->code_align;
      else if ((insn & 0xc0) == DW_CFA_offset)
	{
	  reg = insn & 0x3f;
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  offset = (_Unwind_Sword) utmp * fs->data_align;
	  fs->regs.reg[reg].how = REG_SAVED_OFFSET;
	  fs->regs.reg[reg].loc.offset = offset;
	}
      else if ((insn & 0xc0) == DW_CFA_restore)
	{
	  reg = insn & 0x3f;
	  fs->regs.reg[reg].how = REG_UNSAVED;
	}
      else switch (insn)
	{
	case DW_CFA_set_loc:
	  {
	    _Unwind_Ptr pc;
	    insn_ptr = read_encoded_value (context, fs->fde_encoding,
					   insn_ptr, &pc);
	    fs->pc = (void *) pc;
	  }
	  break;

	case DW_CFA_advance_loc1:
	  fs->pc += read_1u (insn_ptr) * fs->code_align;
	  insn_ptr += 1;
	  break;
	case DW_CFA_advance_loc2:
	  fs->pc += read_2u (insn_ptr) * fs->code_align;
	  insn_ptr += 2;
	  break;
	case DW_CFA_advance_loc4:
	  fs->pc += read_4u (insn_ptr) * fs->code_align;
	  insn_ptr += 4;
	  break;

	case DW_CFA_offset_extended:
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  offset = (_Unwind_Sword) utmp * fs->data_align;
	  fs->regs.reg[reg].how = REG_SAVED_OFFSET;
	  fs->regs.reg[reg].loc.offset = offset;
	  break;

	case DW_CFA_restore_extended:
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  fs->regs.reg[reg].how = REG_UNSAVED;
	  break;

	case DW_CFA_undefined:
	case DW_CFA_same_value:
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  break;

	case DW_CFA_nop:
	  break;

	case DW_CFA_register:
	  {
	    _Unwind_Word reg2;
	    insn_ptr = read_uleb128 (insn_ptr, &reg);
	    insn_ptr = read_uleb128 (insn_ptr, &reg2);
	    fs->regs.reg[reg].how = REG_SAVED_REG;
	    fs->regs.reg[reg].loc.reg = reg2;
	  }
	  break;

	case DW_CFA_remember_state:
	  {
	    struct frame_state_reg_info *new_rs;
	    if (unused_rs)
	      {
		new_rs = unused_rs;
		unused_rs = unused_rs->prev;
	      }
	    else
	      new_rs = __builtin_alloca (sizeof (struct frame_state_reg_info));

	    *new_rs = fs->regs;
	    fs->regs.prev = new_rs;
	  }
	  break;

	case DW_CFA_restore_state:
	  {
	    struct frame_state_reg_info *old_rs = fs->regs.prev;
#ifdef _LIBC
	    if (old_rs == NULL)
	      __libc_fatal ("Invalid DWARF unwind data.\n");
	    else
#endif
	      {
		fs->regs = *old_rs;
		old_rs->prev = unused_rs;
		unused_rs = old_rs;
	      }
	  }
	  break;

	case DW_CFA_def_cfa:
	  insn_ptr = read_uleb128 (insn_ptr, &fs->cfa_reg);
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  fs->cfa_offset = utmp;
	  fs->cfa_how = CFA_REG_OFFSET;
	  break;

	case DW_CFA_def_cfa_register:
	  insn_ptr = read_uleb128 (insn_ptr, &fs->cfa_reg);
	  fs->cfa_how = CFA_REG_OFFSET;
	  break;

	case DW_CFA_def_cfa_offset:
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  fs->cfa_offset = utmp;
	  /* cfa_how deliberately not set.  */
	  break;

	case DW_CFA_def_cfa_expression:
	  fs->cfa_exp = insn_ptr;
	  fs->cfa_how = CFA_EXP;
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  insn_ptr += utmp;
	  break;

	case DW_CFA_expression:
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  fs->regs.reg[reg].how = REG_SAVED_EXP;
	  fs->regs.reg[reg].loc.exp = insn_ptr;
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  insn_ptr += utmp;
	  break;

	  /* From the 2.1 draft.  */
	case DW_CFA_offset_extended_sf:
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  insn_ptr = read_sleb128 (insn_ptr, &stmp);
	  offset = stmp * fs->data_align;
	  fs->regs.reg[reg].how = REG_SAVED_OFFSET;
	  fs->regs.reg[reg].loc.offset = offset;
	  break;

	case DW_CFA_def_cfa_sf:
	  insn_ptr = read_uleb128 (insn_ptr, &fs->cfa_reg);
	  insn_ptr = read_sleb128 (insn_ptr, &fs->cfa_offset);
	  fs->cfa_how = CFA_REG_OFFSET;
	  break;

	case DW_CFA_def_cfa_offset_sf:
	  insn_ptr = read_sleb128 (insn_ptr, &fs->cfa_offset);
	  /* cfa_how deliberately not set.  */
	  break;

	case DW_CFA_GNU_window_save:
	  /* ??? Hardcoded for SPARC register window configuration.
	     At least do not do anything for archs which explicitly
	     define a lower register number.  */
#if DWARF_FRAME_REGISTERS >= 32
	  for (reg = 16; reg < 32; ++reg)
	    {
	      fs->regs.reg[reg].how = REG_SAVED_OFFSET;
	      fs->regs.reg[reg].loc.offset = (reg - 16) * sizeof (void *);
	    }
#endif
	  break;

	case DW_CFA_GNU_args_size:
	  insn_ptr = read_uleb128 (insn_ptr, &context->args_size);
	  break;

	case DW_CFA_GNU_negative_offset_extended:
	  /* Obsoleted by DW_CFA_offset_extended_sf, but used by
	     older PowerPC code.  */
	  insn_ptr = read_uleb128 (insn_ptr, &reg);
	  insn_ptr = read_uleb128 (insn_ptr, &utmp);
	  offset = (_Unwind_Word) utmp * fs->data_align;
	  fs->regs.reg[reg].how = REG_SAVED_OFFSET;
	  fs->regs.reg[reg].loc.offset = -offset;
	  break;

	default:
	  abort ();
	}
    }
}

/* Given the _Unwind_Context CONTEXT for a stack frame, look up the FDE for
   its caller and decode it into FS.  This function also sets the
   args_size and lsda members of CONTEXT, as they are really information
   about the caller's frame.  */

static _Unwind_Reason_Code
uw_frame_state_for (struct _Unwind_Context *context, _Unwind_FrameState *fs)
{
  struct dwarf_fde *fde;
  struct dwarf_cie *cie;
  const unsigned char *aug, *insn, *end;

  memset (fs, 0, sizeof (*fs));
  context->args_size = 0;
  context->lsda = 0;

  fde = _Unwind_Find_FDE (context->ra - 1, &context->bases);
  if (fde == NULL)
    {
      /* Couldn't find frame unwind info for this function.  Try a
	 target-specific fallback mechanism.  This will necessarily
	 not provide a personality routine or LSDA.  */
#ifdef MD_FALLBACK_FRAME_STATE_FOR
      MD_FALLBACK_FRAME_STATE_FOR (context, fs, success);
      return _URC_END_OF_STACK;
    success:
      return _URC_NO_REASON;
#else
      return _URC_END_OF_STACK;
#endif
    }

  fs->pc = context->bases.func;

  cie = get_cie (fde);
  insn = extract_cie_info (cie, context, fs);
  if (insn == NULL)
    /* CIE contained unknown augmentation.  */
    return _URC_FATAL_PHASE1_ERROR;

  /* First decode all the insns in the CIE.  */
  end = (unsigned char *) next_fde ((struct dwarf_fde *) cie);
  execute_cfa_program (insn, end, context, fs);

  /* Locate augmentation for the fde.  */
  aug = (unsigned char *) fde + sizeof (*fde);
  aug += 2 * size_of_encoded_value (fs->fde_encoding);
  insn = NULL;
  if (fs->saw_z)
    {
      _Unwind_Word i;
      aug = read_uleb128 (aug, &i);
      insn = aug + i;
    }
  if (fs->lsda_encoding != DW_EH_PE_omit)
    {
      _Unwind_Ptr lsda;
      aug = read_encoded_value (context, fs->lsda_encoding, aug, &lsda);
      context->lsda = (void *) lsda;
    }

  /* Then the insns in the FDE up to our target PC.  */
  if (insn == NULL)
    insn = aug;
  end = (unsigned char *) next_fde (fde);
  execute_cfa_program (insn, end, context, fs);

  return _URC_NO_REASON;
}

typedef struct frame_state
{
  void *cfa;
  void *eh_ptr;
  long cfa_offset;
  long args_size;
  long reg_or_offset[PRE_GCC3_DWARF_FRAME_REGISTERS+1];
  unsigned short cfa_reg;
  unsigned short retaddr_column;
  char saved[PRE_GCC3_DWARF_FRAME_REGISTERS+1];
} frame_state;

#ifndef STATIC
# define STATIC
#endif

STATIC
struct frame_state * __frame_state_for (void *, struct frame_state *);

/* Called from pre-G++ 3.0 __throw to find the registers to restore for
   a given PC_TARGET.  The caller should allocate a local variable of
   `struct frame_state' and pass its address to STATE_IN.  */

STATIC
struct frame_state *
__frame_state_for (void *pc_target, struct frame_state *state_in)
{
  struct _Unwind_Context context;
  _Unwind_FrameState fs;
  int reg;

  memset (&context, 0, sizeof (struct _Unwind_Context));
  context.ra = pc_target + 1;

  if (uw_frame_state_for (&context, &fs) != _URC_NO_REASON)
    return 0;

  /* We have no way to pass a location expression for the CFA to our
     caller.  It wouldn't understand it anyway.  */
  if (fs.cfa_how == CFA_EXP)
    return 0;

  for (reg = 0; reg < PRE_GCC3_DWARF_FRAME_REGISTERS + 1; reg++)
    {
      state_in->saved[reg] = fs.regs.reg[reg].how;
      switch (state_in->saved[reg])
	{
	case REG_SAVED_REG:
	  state_in->reg_or_offset[reg] = fs.regs.reg[reg].loc.reg;
	  break;
	case REG_SAVED_OFFSET:
	  state_in->reg_or_offset[reg] = fs.regs.reg[reg].loc.offset;
	  break;
	default:
	  state_in->reg_or_offset[reg] = 0;
	  break;
	}
    }

  state_in->cfa_offset = fs.cfa_offset;
  state_in->cfa_reg = fs.cfa_reg;
  state_in->retaddr_column = fs.retaddr_column;
  state_in->args_size = context.args_size;
  state_in->eh_ptr = fs.eh_ptr;

  return state_in;
}

#ifndef _LIBC

static void
uw_update_context_1 (struct _Unwind_Context *context, _Unwind_FrameState *fs)
{
  struct _Unwind_Context orig_context = *context;
  void *cfa;
  long i;

#ifdef EH_RETURN_STACKADJ_RTX
  /* Special handling here: Many machines do not use a frame pointer,
     and track the CFA only through offsets from the stack pointer from
     one frame to the next.  In this case, the stack pointer is never
     stored, so it has no saved address in the context.  What we do
     have is the CFA from the previous stack frame.

     In very special situations (such as unwind info for signal return),
     there may be location expressions that use the stack pointer as well.

     Do this conditionally for one frame.  This allows the unwind info
     for one frame to save a copy of the stack pointer from the previous
     frame, and be able to use much easier CFA mechanisms to do it.
     Always zap the saved stack pointer value for the next frame; carrying
     the value over from one frame to another doesn't make sense.  */

  _Unwind_Word tmp_sp;

  if (!orig_context.reg[__builtin_dwarf_sp_column ()])
    {
      tmp_sp = (_Unwind_Ptr) context->cfa;
      orig_context.reg[__builtin_dwarf_sp_column ()] = &tmp_sp;
    }
  context->reg[__builtin_dwarf_sp_column ()] = NULL;
#endif

  /* Compute this frame's CFA.  */
  switch (fs->cfa_how)
    {
    case CFA_REG_OFFSET:
      cfa = (void *) (_Unwind_Ptr) _Unwind_GetGR (&orig_context, fs->cfa_reg);
      cfa += fs->cfa_offset;
      break;

    case CFA_EXP:
      {
	const unsigned char *exp = fs->cfa_exp;
	_Unwind_Word len;

	exp = read_uleb128 (exp, &len);
	cfa = (void *) (_Unwind_Ptr)
	  execute_stack_op (exp, exp + len, &orig_context, 0);
	break;
      }

    default:
      abort ();
    }
  context->cfa = cfa;

  /* Compute the addresses of all registers saved in this frame.  */
  for (i = 0; i < DWARF_FRAME_REGISTERS + 1; ++i)
    switch (fs->regs.reg[i].how)
      {
      case REG_UNSAVED:
	break;

      case REG_SAVED_OFFSET:
	context->reg[i] = cfa + fs->regs.reg[i].loc.offset;
	break;

      case REG_SAVED_REG:
	context->reg[i] = orig_context.reg[fs->regs.reg[i].loc.reg];
	break;

      case REG_SAVED_EXP:
	{
	  const unsigned char *exp = fs->regs.reg[i].loc.exp;
	  _Unwind_Word len;
	  _Unwind_Ptr val;

	  exp = read_uleb128 (exp, &len);
	  val = execute_stack_op (exp, exp + len, &orig_context,
				  (_Unwind_Ptr) cfa);
	  context->reg[i] = (void *) val;
	}
	break;
      }
}

/* CONTEXT describes the unwind state for a frame, and FS describes the FDE
   of its caller.  Update CONTEXT to refer to the caller as well.  Note
   that the args_size and lsda members are not updated here, but later in
   uw_frame_state_for.  */

static void
uw_update_context (struct _Unwind_Context *context, _Unwind_FrameState *fs)
{
  uw_update_context_1 (context, fs);

  /* Compute the return address now, since the return address column
     can change from frame to frame.  */
  context->ra = __builtin_extract_return_addr
    ((void *) (_Unwind_Ptr) _Unwind_GetGR (context, fs->retaddr_column));
}

/* Fill in CONTEXT for top-of-stack.  The only valid registers at this
   level will be the return address and the CFA.  */

#define uw_init_context(CONTEXT)					   \
  do									   \
    {									   \
      /* Do any necessary initialization to access arbitrary stack frames. \
	 On the SPARC, this means flushing the register windows.  */	   \
      __builtin_unwind_init ();						   \
      uw_init_context_1 (CONTEXT, __builtin_dwarf_cfa (),		   \
			 __builtin_return_address (0));			   \
    }									   \
  while (0)

static void
uw_init_context_1 (struct _Unwind_Context *context,
		   void *outer_cfa, void *outer_ra)
{
  void *ra = __builtin_extract_return_addr (__builtin_return_address (0));
  _Unwind_FrameState fs;
  _Unwind_Word sp_slot;

  memset (context, 0, sizeof (struct _Unwind_Context));
  context->ra = ra;

  if (uw_frame_state_for (context, &fs) != _URC_NO_REASON)
    abort ();

  /* Force the frame state to use the known cfa value.  */
  sp_slot = (_Unwind_Ptr) outer_cfa;
  context->reg[__builtin_dwarf_sp_column ()] = &sp_slot;
  fs.cfa_how = CFA_REG_OFFSET;
  fs.cfa_reg = __builtin_dwarf_sp_column ();
  fs.cfa_offset = 0;

  uw_update_context_1 (context, &fs);

  /* If the return address column was saved in a register in the
     initialization context, then we can't see it in the given
     call frame data.  So have the initialization context tell us.  */
  context->ra = __builtin_extract_return_addr (outer_ra);
}


/* Install TARGET into CURRENT so that we can return to it.  This is a
   macro because __builtin_eh_return must be invoked in the context of
   our caller.  */

#define uw_install_context(CURRENT, TARGET)				 \
  do									 \
    {									 \
      long offset = uw_install_context_1 ((CURRENT), (TARGET));		 \
      void *handler = __builtin_frob_return_addr ((TARGET)->ra);	 \
      __builtin_eh_return (offset, handler);				 \
    }									 \
  while (0)

static inline void
init_dwarf_reg_size_table (void)
{
  __builtin_init_dwarf_reg_size_table (dwarf_reg_size_table);
}

static long
uw_install_context_1 (struct _Unwind_Context *current,
		      struct _Unwind_Context *target)
{
  long i;

#if __GTHREADS
  {
    static __gthread_once_t once_regsizes = __GTHREAD_ONCE_INIT;
    if (__gthread_once (&once_regsizes, init_dwarf_reg_size_table) != 0
	|| dwarf_reg_size_table[0] == 0)
      init_dwarf_reg_size_table ();
  }
#else
  if (dwarf_reg_size_table[0] == 0)
    init_dwarf_reg_size_table ();
#endif

  for (i = 0; i < DWARF_FRAME_REGISTERS; ++i)
    {
      void *c = current->reg[i];
      void *t = target->reg[i];
      if (t && c && t != c)
	memcpy (c, t, dwarf_reg_size_table[i]);
    }

#ifdef EH_RETURN_STACKADJ_RTX
  {
    void *target_cfa;

    /* If the last frame records a saved stack pointer, use it.  */
    if (target->reg[__builtin_dwarf_sp_column ()])
      target_cfa = (void *)(_Unwind_Ptr)
        _Unwind_GetGR (target, __builtin_dwarf_sp_column ());
    else
      target_cfa = target->cfa;

    /* We adjust SP by the difference between CURRENT and TARGET's CFA.  */
    if (STACK_GROWS_DOWNWARD)
      return target_cfa - current->cfa + target->args_size;
    else
      return current->cfa - target_cfa - target->args_size;
  }
#else
  return 0;
#endif
}

static inline _Unwind_Ptr
uw_identify_context (struct _Unwind_Context *context)
{
  return _Unwind_GetIP (context);
}


#include "unwind.inc"

#endif /* _LIBC */
