/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "gbldefs.h"
#include "error.h"
#include "ll_structure.h"
#include "ll_write.h"
#include "global.h"
#include "mach.h"
#include "dwarf2.h"
#include "llutil.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef TARGET_LLVM_ARM64
#include "cgllvm.h"
#endif

#ifdef OMP_OFFLOAD_LLVM
#include "ll_builder.h"
#endif

#define SPACES "    "

#ifdef TARGET_POWER
#define POWER_STACK_32_BIT_NAN "2146959359" /* 0x7FF7FFFF */
/* Two consecutive 32 bit NaNs form a 64 bit SNan*/
#define POWER_STACK_64_BIT_NAN "9221120234893082623" /* 0x7FF7FFFF7FF7FFFF */
#endif

static LL_Function *called;
static int debug_calls = 0;
static int text_calls = 0;
static const char *ll_get_atomic_memorder(LL_Instruction *inst);

static const char *
ll_get_linkage_string(enum LL_LinkageType linkage)
{
  switch (linkage) {
  case LL_INTERNAL_LINKAGE:
    return "internal";
  case LL_COMMON_LINKAGE:
    return "common";
  case LL_WEAK_LINKAGE:
    return "weak";
  case LL_EXTERNAL_LINKAGE:
    return "external";
  case LL_NO_LINKAGE:
    break;
  }
  return "";
}

/**
   \brief Write out the header of a module.

   The header is: Module ID, target triple, and datalayout
 */
void
ll_write_module_header(FILE *out, LLVMModuleRef module)
{
  if (module->module_name[0])
    fprintf(out, "; ModuleID = '%s'\n", module->module_name);
  if (module->datalayout_string[0])
    fprintf(out, "target datalayout = \"%s\"\n", module->datalayout_string);
  if (module->target_triple[0])
    fprintf(out, "target triple = \"%s\"\n", module->target_triple);
}

/**
   \brief Write out definitions for named struct types in module.

   If this function is called more than once, only the new types added since
   the last call are written.
 */
void
ll_write_user_structs(FILE *out, LLVMModuleRef module)
{
  unsigned i, j;

  for (i = module->written_user_structs; i < module->num_user_structs; i++) {
    LL_Value *val = module->user_structs.values[i];
    LL_Type *type = val->type_struct;
    int packed = type->flags & LL_TYPE_IS_PACKED_STRUCT;

    /* TODO: Should we support opaque struct types too? */
    if (type->sub_elements == 0) {
      fprintf(out, "%s = type %s\n", type->str, packed ? "<{}>" : "{}");
      continue;
    }

    fprintf(out, "%s = type %s{%s", type->str, packed ? "<" : "",
            type->sub_types[0]->str);
    for (j = 1; j < type->sub_elements; j++)
      fprintf(out, ", %s", type->sub_types[j]->str);
    fprintf(out, "}%s\n", packed ? ">" : "");
  }

  /* Avoid rewriting these structs if called again. */
  module->written_user_structs = i;
}

static const char *
get_op_name(enum LL_Op op)
{
  switch (op) {
  case LL_FPTRUNC:
    return "fptrunc";
  case LL_FPEXT:
    return "fpext";
  case LL_SEXT:
    return "sext";
  case LL_ZEXT:
    return "zext";
  case LL_TRUNC:
    return "trunc";
  case LL_BITCAST:
    return "bitcast";
  case LL_SITOFP:
    return "sitofp";
  case LL_UITOFP:
    return "uitofp";
  case LL_FPTOSI:
    return "fptosi";
  case LL_FPTOUI:
    return "fptoui";
  case LL_ADD:
    return "add";
  case LL_FADD:
    return "fadd";
  case LL_SUB:
    return "sub";
  case LL_FSUB:
    return "fsub";
  case LL_MUL:
    return "mul";
  case LL_FMUL:
    return "fmul";
  case LL_UDIV:
    return "udiv";
  case LL_SDIV:
    return "sdiv";
  case LL_SREM:
    return "srem";
  case LL_UREM:
    return "urem";
  case LL_FDIV:
    return "fdiv";
  case LL_OR:
    return "or";
  case LL_ASHR:
    return "ashr";
  case LL_LSHR:
    return "lshr";
  case LL_AND:
    return "and";
  case LL_XOR:
    return "xor";
  case LL_SHL:
    return "shl";
  case LL_INTTOPTR:
    return "inttoptr";
  case LL_PTRTOINT:
    return "ptrtoint";
  case LL_ICMP:
    return "icmp";
  case LL_FCMP:
    return "fcmp";
  case LL_ATOMICRMW:
    return "atomicrmw";
  case LL_CMPXCHG:
    return "cmpxchg";
  case LL_EXTRACTVALUE:
    return "extractvalue";
  case LL_INSERTVALUE:
    return "insertvalue";
  case LL_EXTRACTELEM:
    return "extractelement";
  case LL_INSERTELEM:
    return "insertelement";
  case LL_FNEG:
    return "fneg";
  default:
    return "thisisnotacceptable";
  }
}

static void
add_prototype(LL_Instruction *instruction)
{
  LL_Function *scan_function = called;
  LL_Function *new_function;
  LL_Value *function = instruction->operands[1];
  LL_Value *attributes = function->storage;
  int i;

  if (function->data == NULL) {
    fprintf(stderr, "Attempting to add prototype for function with no name.\n");
    return;
  }

  while (scan_function != NULL) {
    if (strcmp(scan_function->name, function->data) == 0) {
      /* We've already prototyped this function.  Exit. */
      return;
    }
    scan_function = scan_function->next;
  }
  new_function = (LL_Function *)malloc(sizeof(LL_Function));
  ll_set_function_num_arguments(new_function, instruction->num_operands - 2);
  new_function->next = called;
  new_function->name = function->data;
  new_function->num_args = instruction->num_operands - 2;
  new_function->attributes = (attributes ? attributes->data : NULL);

  new_function->return_type = function->type_struct;
  called = new_function;

  for (i = 2; i < instruction->num_operands; i++) {
    new_function->arguments[i - 2] = instruction->operands[i];
  }
}

static bool
defined_in_module(LL_Function *function, LLVMModuleRef module)
{
  LL_Function *scan_function;
  scan_function = module->first;
  while (scan_function != NULL) {
    if (strcmp(scan_function->name, function->name) == 0)
      return true;
    scan_function = scan_function->next;
  }
  return false;
}

static void
write_prototypes(FILE *out, LLVMModuleRef module)
{
  LL_Function *cur_function = called;

  while (cur_function != NULL) {
    if (!defined_in_module(cur_function, module)) {
      fprintf(out, "\ndeclare %s @%s(", cur_function->return_type->str,
              cur_function->name);
      for (unsigned int i = 0; i < cur_function->num_args; i++) {
        fprintf(out, "%s", cur_function->arguments[i]->type_struct->str);

        if (i + 1 < cur_function->num_args) {
          fprintf(out, ", ");
        }
      }
      fprintf(out, ") nounwind");
      if (cur_function->attributes) {
        fprintf(out, " %s", cur_function->attributes);
      }
    }
    cur_function = cur_function->next;
  }
  fprintf(out, "\n");
  if (debug_calls) {
    fprintf(out, "declare void @llvm.dbg.declare(metadata, metadata) nounwind "
                 "readnone\n");
    fprintf(out, "declare void @llvm.dbg.value(metadata, i64, metadata) "
                 "nounwind readnone\n\n");
  }

  if (text_calls) {
    fprintf(out, "declare i64 @llvm.nvvm.texsurf.handle.p1i64(metadata, "
                 "ptr addrspace(1)) nounwind readnone\n");
  }

  for (int i = 0; i < module->num_refs; i++) {
    if (module->extern_func_refs[i] != NULL)
      fprintf(out, "declare void %s()\n", module->extern_func_refs[i]->data);
  }
}

static void
clear_prototypes(void)
{
  LL_Function *scan_function = called;
  LL_Function *next_function;

  while (scan_function != NULL) {
    free(scan_function->arguments);
    next_function = scan_function->next;
    free(scan_function);
    scan_function = next_function;
  }
  called = NULL;
}

static void
render_bitcast(FILE *out, LL_Instruction *inst)
{
  const char *cast_operand = inst->operands[1]->data;

  if (inst->operands[1]->type_struct->data_type == LL_PTR &&
      strcmp(inst->operands[1]->data, "0") == 0) {
    /* Replace "0" with "null" */
    cast_operand = "null";
  }
  fprintf(out, "%s%s = bitcast %s %s to %s", SPACES, inst->operands[0]->data,
          inst->operands[1]->type_struct->str, cast_operand,
          inst->operands[0]->type_struct->str);
}

static void
render_store(FILE *out, LL_Instruction *inst)
{
  const char *store_operand = inst->operands[0]->data;
  char szatomic[25];
  char szmemorder[128];
  sprintf(szatomic, "");
  sprintf(szmemorder, "");

  if (inst->operands[0]->type_struct->data_type == LL_PTR &&
      strcmp(inst->operands[0]->data, "0") == 0) {
    /* Replace "0" with "null" */
    store_operand = "null";
  } else if (inst->operands[0]->type_struct->data_type == LL_FLOAT) {
    if (strcmp(inst->operands[0]->data, "inf") == 0) {
      store_operand = "0x7FF0000000000000";
    } else if (strcmp(inst->operands[0]->data, "-inf") == 0) {
      store_operand = "0xFFF0000000000000";
    } else if (strcmp(inst->operands[0]->data, "nan") == 0) {
      store_operand = "0x7FF8000000000000";
    }
  }
  fprintf(out, "%sstore%s%s %s %s, %s %s %s", SPACES, szatomic,
          (inst->flags & INST_VOLATILE) ? " volatile" : "",
          inst->operands[0]->type_struct->str, store_operand,
          inst->operands[1]->type_struct->str, inst->operands[1]->data,
          szmemorder);

  if (inst->num_operands >= 3)
    fprintf(out, ", align %s", inst->operands[2]->data);
}

static const char *szatomic_opr[10] = {"none", "xchg", "add", "sub", "and",
                                       "nand", "or",   "xor", "max", "min"};
static const char *
ll_get_atomic_opr(LL_Instruction *inst)
{
  int flags = (inst->flags & ATOMIC_RMW_OP_FLAGS);
  const char *szopr = NULL;
  int idx = flags >> 13;

  switch (flags) {
  case ATOMIC_XCHG_FLAG:
  case ATOMIC_ADD_FLAG:
  case ATOMIC_SUB_FLAG:
  case ATOMIC_AND_FLAG:
  case ATOMIC_OR_FLAG:
  case ATOMIC_XOR_FLAG:
  case ATOMIC_MIN_FLAG:
  case ATOMIC_MAX_FLAG:
    szopr = szatomic_opr[idx];
    break;
  default:
    assert(false, "unimplemented op in ll_get_atomic_opr", flags, ERR_Fatal);
  }
  return szopr;
}

static const char *szmemorder[7] = {"undef",   "monotonic", "undef",  "acquire",
                                    "release", "acq_rel",   "seq_cst"};
static const char *
ll_get_atomic_memorder(LL_Instruction *inst)
{
  int instr_flags = inst->flags;
  int idx = (instr_flags & ATOMIC_MEM_ORD_FLAGS) >> 18;
  const char *memorder = NULL;
  switch (instr_flags & ATOMIC_MEM_ORD_FLAGS) {
  case ATOMIC_MONOTONIC_FLAG:
  case ATOMIC_ACQUIRE_FLAG:
  case ATOMIC_RELEASE_FLAG:
  case ATOMIC_ACQ_REL_FLAG:
  case ATOMIC_SEQ_CST_FLAG:
    memorder = szmemorder[idx];
    break;
  default:
    interr("Unexpected atomic mem ord flag: ",
           instr_flags & ATOMIC_MEM_ORD_FLAGS, ERR_Severe);
  }
  return memorder;
}

void
ll_write_instruction(FILE *out, LL_Instruction *inst, LL_Module *module, int no_return)
{
  const char *opname;
  int i;
  int print_branch_target;

  if (inst->flags & INST_CANCELED)
    return;
  print_branch_target = 0;
  opname = get_op_name(inst->op);
  switch (inst->op) {
  case LL_ASM: {
    if(inst->num_operands==2) {
      fprintf(out, "%scall void asm sideeffect \"%s\", \"\"()", SPACES,
            inst->operands[1]->data);
    } else {
      int noperands = inst->num_operands;
      if(inst->operands[0]->type_struct->data_type!=LL_VOID)
        fprintf(out, "%s%s = ", SPACES, inst->operands[0]->data);
      else 
        fprintf(out, "%s", SPACES);
      fprintf(out, "call %s asm sideeffect \"%s\", \"%s\"", 
            inst->operands[0]->type_struct->str, 
            inst->operands[1]->data, inst->operands[2]->data);
      fprintf(out, "(");
      for(i=3; i<noperands; i++) {
        fprintf(out, "%s %s", inst->operands[i]->type_struct->str, inst->operands[i]->data);
        if(i<(noperands-1))
          fprintf(out, ",");
      }
      fprintf(out, ")");
    }
  }
  break;
  case LL_ATOMICRMW: {
    const char *atomicopr;
    const char *memorder;
    atomicopr = ll_get_atomic_opr(inst);
    memorder = ll_get_atomic_memorder(inst);
    fprintf(out, "%s%s = %s %s %s %s, %s %s %s", SPACES,
            inst->operands[0]->data, opname, atomicopr,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            memorder);

  } break;
  case LL_CMPXCHG: {
    const char *memorder;
    memorder = ll_get_atomic_memorder(inst);
    fprintf(out, "%s%s = %s %s %s, %s %s, %s %s %s", SPACES,
            inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            inst->operands[3]->type_struct->str, inst->operands[3]->data,
            memorder);
  } break;
  case LL_EXTRACTVALUE: {
    fprintf(out, "%s%s = %s %s %s, %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->data);
  } break;
  case LL_INSERTVALUE: {
    fprintf(out, "%s%s = %s %s %s, %s %s, %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            inst->operands[3]->data);
  } break;
  case LL_EXTRACTELEM: {
    fprintf(out, "%s%s = %s %s %s, %s %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data);
  } break;
  case LL_INSERTELEM: {
    fprintf(out, "%s%s = %s %s %s, %s %s, %s %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            inst->operands[3]->type_struct->str, inst->operands[3]->data);
  } break;
  case LL_ADD:
  case LL_FADD:
  case LL_SUB:
  case LL_FSUB:
  case LL_MUL:
  case LL_FMUL:
  case LL_UDIV:
  case LL_SDIV:
  case LL_FDIV:
  case LL_UREM:
  case LL_SREM:
  case LL_ASHR:
  case LL_OR:
  case LL_AND:
  case LL_XOR:
  case LL_LSHR:
  case LL_SHL:
    /* Group all binary operations */
    fprintf(out, "%s%s = %s %s %s, %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data,
            inst->operands[2]->data);
    break;
  case LL_FNEG:
    /* unary ops */
    fprintf(out, "%s%s = %s %s %s", SPACES, inst->operands[0]->data, opname,
            inst->operands[1]->type_struct->str, inst->operands[1]->data);
    break;
  case LL_STORE:
    render_store(out, inst);
    break;
  case LL_LOAD:
    fprintf(out, "%s%s = load %s", SPACES, inst->operands[0]->data,
            (inst->flags & INST_VOLATILE) ? "volatile " : "");
    if (ll_feature_explicit_gep_load_type(&module->ir))
      fprintf(out, "%s, ", inst->operands[1]->type_struct->sub_types[0]->str);
    fprintf(out, "%s %s", inst->operands[1]->type_struct->str,
            inst->operands[1]->data);
    if (inst->num_operands >= 3)
      fprintf(out, ", align %s", inst->operands[2]->data);
    break;
  case LL_SEXT:
  case LL_ZEXT:
  case LL_TRUNC:
  case LL_FPTRUNC:
  case LL_FPEXT:
  case LL_SITOFP:
  case LL_UITOFP:
  case LL_PTRTOINT:
  case LL_INTTOPTR:
  case LL_FPTOSI:
  case LL_FPTOUI:
    /* Group all conversion operations */
    fprintf(out, "%s%s = %s %s %s to %s", SPACES, inst->operands[0]->data,
            opname, inst->operands[1]->type_struct->str,
            inst->operands[1]->data, inst->operands[0]->type_struct->str);
    break;
  case LL_BITCAST:
    render_bitcast(out, inst);
    break;
  case LL_RET:
    if (no_return) {
      fprintf(out, "%scall void @llvm.nvvm.exit()\n",SPACES);
      fprintf(out, "%sunreachable",SPACES);
    }
    else
      fprintf(out, "%sret %s %s", SPACES, inst->operands[0]->type_struct->str,
                    inst->operands[0]->data);
    break;
  case LL_ICMP:
  case LL_FCMP:
    fprintf(out, "%s%s = %s %s %s %s, %s", SPACES, inst->operands[0]->data,
            opname, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            inst->operands[3]->data);
    break;
  case LL_SELECT:
    fprintf(out, "%s%s = select i1 %s, %s %s, %s %s", SPACES,
            inst->operands[0]->data, inst->operands[1]->data,
            inst->operands[2]->type_struct->str, inst->operands[2]->data,
            inst->operands[3]->type_struct->str, inst->operands[3]->data);
    break;
  case LL_BR:
    fprintf(out, "%sbr i1 %s, label %%%s, label %%%s", SPACES,
            inst->operands[0]->data, inst->operands[1]->data,
            inst->operands[2]->data);
    print_branch_target = 1;
    break;
  case LL_UBR:
    fprintf(out, "%sbr label %%%s", SPACES, inst->operands[0]->data);
    break;
  case LL_CALL:
    if (inst->flags & INST_INDIRECT_CALL) {
      if (inst->operands[0]->type_struct->data_type != LL_VOID) {
        fprintf(out, "%s%s = call %s %s(", SPACES, inst->operands[0]->data,
                inst->operands[0]->type_struct->str, inst->operands[1]->data);
      } else {
        fprintf(out, "%scall %s %s(", SPACES,
                inst->operands[0]->type_struct->str, inst->operands[1]->data);
      }
    } else {
      if (inst->operands[0]->type_struct->data_type != LL_VOID) {
        fprintf(out, "%s%s = call %s @%s(", SPACES, inst->operands[0]->data,
                inst->operands[1]->type_struct->str, inst->operands[1]->data);
      } else {
        fprintf(out, "%scall %s @%s(", SPACES,
                inst->operands[1]->type_struct->str, inst->operands[1]->data);
      }
    }
    for (i = 2; i < inst->num_operands; i++) {
      fprintf(out, "%s %s", inst->operands[i]->type_struct->str,
              inst->operands[i]->data);
      if (i + 1 < inst->num_operands) {
        fprintf(out, ", ");
      }
    }
    fprintf(out, ")");
    if (!(inst->flags & (IN_MODULE_CALL | INST_INDIRECT_CALL))) {
      add_prototype(inst);
    }
    break;
  case LL_TEXTCALL:
    if (inst->operands[0]->type_struct->data_type != LL_VOID) {
      fprintf(out, "%s%s = call %s @%s(", SPACES, inst->operands[0]->data,
              inst->operands[1]->type_struct->str, inst->operands[1]->data);
    } else {
      fprintf(out, "%scall %s @%s(", SPACES,
              inst->operands[1]->type_struct->str, inst->operands[1]->data);
    }
    fprintf(out, "metadata !{%s %s}, ", inst->operands[2]->type_struct->str,
            inst->operands[2]->data);
    fprintf(out, "%s %s", inst->operands[2]->type_struct->str,
            inst->operands[2]->data);
    fprintf(out, ")");
    text_calls = 1;
    break;
  case LL_GEP:
    fprintf(out, "%s%s = getelementptr ", SPACES, inst->operands[0]->data);
    if(inst->flags&INST_INBOUND)
      fprintf(out, "inbounds ");
    if (ll_feature_explicit_gep_load_type(&module->ir))
      fprintf(out, "%s, ", inst->operands[1]->type_struct->sub_types[0]->str);
    fprintf(out, "%s %s", inst->operands[1]->type_struct->str,
            inst->operands[1]->data);
    for (i = 2; i < inst->num_operands; i++) {
      fprintf(out, ", %s %s", inst->operands[i]->type_struct->str,
              inst->operands[i]->data);
    }
    break;
  case LL_ALLOCA:
    fprintf(out, "%s%s = alloca %s", SPACES, inst->operands[0]->data,
            inst->operands[1]->type_struct->str);
    /* alloca size */
    if (inst->num_operands >= 4)
      fprintf(out, ", %s %s", inst->operands[3]->type_struct->str, inst->operands[3]->data);
    /* alignment */
    if (inst->num_operands >= 3)
      fprintf(out, ", align %s", inst->operands[2]->data);
    break;
  case LL_UNREACHABLE:
    fprintf(out, "%sunreachable", SPACES);
    break;
  case LL_SWITCH:
    fprintf(out, "%sswitch %s %s, label %%%s [\n", SPACES,
            inst->operands[0]->type_struct->str, inst->operands[0]->data,
            inst->operands[1]->data);
    for (i = 2; i < inst->num_operands; i += 2) {
      fprintf(out, "%s  %s %s, label %%%s\n", SPACES,
              inst->operands[i + 0]->type_struct->str,
              inst->operands[i + 0]->data, inst->operands[i + 1]->data);
    }
    fprintf(out, "%s]", SPACES);
    break;
  case LL_NONE:
    break;
  }
  if (!LL_MDREF_IS_NULL(inst->dbg_line_op)) {
    fprintf(out, ", !dbg !%u", LL_MDREF_value(inst->dbg_line_op));
  }
#if DEBUG
  if (inst->comment)
    fprintf(out, " ; %s", inst->comment);
#endif

  fputc('\n', out);
  if (print_branch_target)
    fprintf(out, "%s:\n", inst->operands[2]->data);
  fflush(out);
}

/**
   \brief Emit a list of \c !dbg \e n annotations
   \param ods  the object to \c !dbg list

   In LLVM 4.0, we can generate a list of comma separated \c !dbg metadata to
   link the object to a number of debug metadata descriptions.
 */
void
ll_write_object_dbg_references(FILE *out, LL_Module *m, LL_ObjToDbgList *ods)
{
  LL_ObjToDbgListIter i;
  if (!ll_feature_from_global_to_md(&m->ir))
    return;
  for (llObjtodbgFirst(ods, &i); !llObjtodbgAtEnd(&i); llObjtodbgNext(&i)) {
    LL_MDRef mdnode = llObjtodbgGet(&i);
    fprintf(out, ", !dbg !%u", LL_MDREF_value(mdnode));
  }
  llObjtodbgFree(ods);
}

void
ll_write_basicblock(FILE *out, LL_Function *function, LL_BasicBlock *block,
                    LL_Module *module, int no_return)
{
  LL_Instruction *inst = block->first;

  if (block->name)
    fprintf(out, "%s:\n", block->name);

  if (block == function->first)
    ll_write_local_objects(out, function);

  while (inst) {
    ll_write_instruction(out, inst, module, no_return);
    inst = inst->next;
  }
}

/**
   \brief Write out definitions of local objects in function as a series of
   alloca instructions.

   Unlike ll_write_global_objects(), this function only expects to be called
   once per function.
 */
void
ll_write_local_objects(FILE *out, LL_Function *function)
{
  LL_Object *object;
#ifdef TARGET_POWER
  int i;
  int curr_nan_label_count = 0;
  const char *name;
#endif

  for (object = function->first_local; object; object = object->next) {
    fprintf(out, "\t%s = alloca %s", object->address.data, object->type->str);
    if (object->align_bytes)
      fprintf(out, ", align %u", object->align_bytes);
    fputc('\n', out);

#ifdef TARGET_LLVM_ARM64
    // See process_formal_arguments in cgmain.c for handling on ARM64 of locals
    // smaller than formals
    if (object->kind == LLObj_LocalBuffered) {
      assert(object->sptr && strcmp(object->address.data, SNAME(object->sptr)), 
              "Missing local storage", object->sptr, ERR_Fatal);
      LL_Type * llt = make_lltype_from_sptr((SPTR)object->sptr);
      fprintf(out, "\t%s = bitcast ptr %s to %s",
              SNAME(object->sptr), object->address.data, llt->str);
      fputc('\n', out);
    }
#endif

#ifdef TARGET_POWER
    if (XBIT(217, 0x1)) {
      name = object->address.data;
      if (ll_type_bytes(object->type) == 4) {
        fprintf(out, "\tstore i32 %s, ptr %s, align 4\n",
                POWER_STACK_32_BIT_NAN, name);
      } else if (ll_type_bytes(object->type) == 8) {
        fprintf(out, "\tstore i64 %s, ptr %s, align 8\n",
                POWER_STACK_64_BIT_NAN, name);
      } else if (ll_type_bytes(object->type) > 4) {
        fprintf(out, "\t%s.ptr = alloca ptr, align 4\n", name);
        fprintf(out, "\t%s.count = alloca i32, align 4\n", name);
        fprintf(out, "\tstore i32 %d, ptr %s.count, align 4\n",
                (int)(ll_type_bytes(object->type) / 4), name);
        fprintf(out, "\tstore ptr %s, ptr %s.ptr, align 4\n", name,
                name);
        fprintf(out, "\tbr label %%L.st.init.%04d.1\n", curr_nan_label_count);
        fprintf(out, "L.st.init.%04d.1:\n", curr_nan_label_count);
        fprintf(out, "\t%s.temp = load i32, ptr %s.count, align 4\n", name,
                name);
        fprintf(out, "\t%s.temp0 = icmp sle i32 %s.temp, 0\n", name, name);
        fprintf(out,
                "\tbr i1 %s.temp0, label %%L.st.init.%04d.0,"
                " label %%L.st.init.%04d.2\n",
                name, curr_nan_label_count + 1, curr_nan_label_count);
        fprintf(out, "L.st.init.%04d.2:\n", curr_nan_label_count);
        fprintf(out, "\t%s.temp1 = load ptr, ptr %s.ptr, align 4\n", name,
                name);
        fprintf(out, "\tstore i32 %s, ptr %s.temp1, align 4\n",
                POWER_STACK_32_BIT_NAN, name);
        fprintf(out, "\t%s.temp2 = getelementptr i8, ptr %s.temp1, i32 4\n",
                name, name);
        fprintf(out, "\tstore ptr %s.temp2, ptr %s.ptr, align 4\n", name,
                name);
        fprintf(out, "\t%s.temp3 = load i32, ptr %s.count, align 4\n", name,
                name);
        fprintf(out, "\t%s.temp4 = sub i32 %s.temp3, 1\n", name, name);
        fprintf(out, "\tstore i32 %s.temp4, ptr %s.count, align 4\n", name,
                name);
        fprintf(out, "\tbr label %%L.st.init.%04d.1\n", curr_nan_label_count);
        curr_nan_label_count++;
        fprintf(out, "L.st.init.%04d.0:\n", curr_nan_label_count);
      }
    }
#endif
  }
}

void
ll_write_function(FILE *out, LL_Function *function, LL_Module *module, bool no_return, const char *prefix)
{
  LL_BasicBlock *block = function->first;

  fprintf(out, "define %s %s %s ", ll_get_linkage_string(function->linkage),
          function->calling_convention, function->return_type->str);
  fprintf(out, "@%s%s(", prefix, function->name);
  for (unsigned int i = 0; i < function->num_args; i++) {
    fputs(function->arguments[i]->type_struct->str, out);

    if (function->arguments[i]->flags & VAL_IS_NOALIAS_PARAM) {
      fputs(" noalias", out);
    }

    fputc(' ', out);
    fputs(function->arguments[i]->data, out);
    if (i + 1 < function->num_args)
      fputs(", ", out);
  }
  fputs(") nounwind ", out);
  if (no_return)
    fputs("noreturn ", out);
  fputs("{\n", out);

  while (block) {
    ll_write_basicblock(out, function, block, module, no_return);
    block = block->next;
  }
  fputs("}\n\n", out);
}

/*
 * Metadata
 */

enum FieldType {
  UnsignedField,
  SignedField,
  BoolField,
  NodeField,
  StringField,
  ValueField,
  DWTagField,
  DWLangField,
  DWVirtualityField,
  DWEncodingField,
  DWEmissionField,
  SignedOrMDField,
  DebugNameTableKindField
};

enum FieldFlags {
  FlgMandatory = 0x1, /**< Field must be present, even with a default value */
  FlgOptional = 0x2,  /**< Field does not have to be present in the MDNode */
  FlgHidden = 0x4,    /**< Field is never printed */
  FlgSkip1 = 0x8,     /**< special skip handling of signed constant */
};

/**
 * \brief Templates for printing specialized metadata nodes.
 *
 * A template is an array of MDTemplate structs with nf+1 elements where nf is
 * the number of fields.
 *
 * The first entry of the array is the class name without the leading bang
 * (e.g., "MDLocation"), and the "flags" field contains nf.
 *
 * The remaining entries correspond to the node elements, the names are field
 * names without the trailing colon.
 */
typedef struct MDTemplate {
  const char *name;
  enum FieldType type;
  unsigned flags;
} MDTemplate;

#define TF ((enum FieldType)0)

/* clang-format off */

/* !DILocation(line: 2900, column: 42, scope: !1, inlinedAt: !2) */
static const MDTemplate Tmpl_DILocation[] = {
  { "DILocation", TF, 4 },
  { "line",                     UnsignedField, 0 },
  { "column",                   UnsignedField, 0 },
  { "scope",                    NodeField,     FlgMandatory },
  { "inlinedAt",                NodeField,     0}
};

/* !MDLocation(line: 2900, column: 42, scope: !1, inlinedAt: !2) */
static const MDTemplate Tmpl_MDLocation[] = {
  { "MDLocation", TF, 4 },
  { "line",                     UnsignedField, 0 },
  { "column",                   UnsignedField, 0 },
  { "scope",                    NodeField,     FlgMandatory },
  { "inlinedAt",                NodeField,     0 }
};

/* An DIFile(filename: "...", directory: "...") pair */
static const MDTemplate Tmpl_DIFile_pair[] = {
  { "DIFile", TF, 2 },
  { "filename",                 StringField,   0 },
  { "directory",                StringField,   0 }
};

/* A tagged MDFile node. Not used by LLVM. */
static const MDTemplate Tmpl_DIFile_tagged[] = {
  { "DIFile", TF, 2 },
  { "tag",                      DWTagField, 0 },
  { "pair",                     NodeField,  0 }
};

/* MDFile before 3.4 */
static const MDTemplate Tmpl_DIFile_pre34[] = {
  { "DIFile", TF, 4 },
  { "tag",                      DWTagField,  0 },
  { "filename",                 StringField, 0 },
  { "directory",                StringField, 0 },
  { "context",                  NodeField,   0 }
};

static const MDTemplate Tmpl_DICompileUnit[] = {
  { "DICompileUnit", TF, 13 },
  { "tag",                      DWTagField,    FlgHidden },
  { "file",                     NodeField,     0 },
  { "language",                 DWLangField,   0 },
  { "producer",                 StringField,   0 },
  { "isOptimized",              BoolField,     0 },
  { "flags",                    StringField,   0 },
  { "runtimeVersion",           UnsignedField, 0 },
  { "enums",                    NodeField,     0 },
  { "retainedTypes",            NodeField,     0 },
  { "subprograms",              NodeField,     0 },
  { "globals",                  NodeField,     0 },
  { "imports",                  NodeField,     0 },
  { "splitDebugFilename",       StringField,   0 }
};

/* "subprograms" removed from DICompileUnit in LLVM 3.9 */
static const MDTemplate Tmpl_DICompileUnit_ver39[] = {
  { "DICompileUnit", TF, 14 },
  { "tag",                      DWTagField,              FlgHidden },
  { "file",                     NodeField,               0 },
  { "language",                 DWLangField,             0 },
  { "producer",                 StringField,             0 },
  { "isOptimized",              BoolField,               0 },
  { "flags",                    StringField,             0 },
  { "runtimeVersion",           UnsignedField,           0 },
  { "enums",                    NodeField,               0 },
  { "retainedTypes",            NodeField,               0 },
  { "globals",                  NodeField,               0 },
  { "emissionKind",             DWEmissionField,         0 },
  { "imports",                  NodeField,               0 },
  { "splitDebugFilename",       StringField,             0 },
  { "nameTableKind",            DebugNameTableKindField, 0 }
};

static const MDTemplate Tmpl_DICompileUnit_pre34[] = {
  { "DICompileUnit", TF, 14 },
  { "tag",                      DWTagField,    FlgHidden },
  { "unused",                   NodeField,     FlgHidden },
  { "language",                 DWLangField,   0 },
  { "filename",                 StringField,   0 },
  { "directory",                StringField,   0 },
  { "producer",                 StringField,   0 },
  { "isMain",                   BoolField,     0 },
  { "isOptimized",              BoolField,     0 },
  { "flags",                    StringField,   0 },
  { "runtimeVersion",           UnsignedField, 0 },
  { "enums",                    NodeField,     0 },
  { "retainedTypes",            NodeField,     0 },
  { "subprograms",              NodeField,     0 },
  { "globals",                  NodeField,     0 }
};

static const MDTemplate Tmpl_DINamespace_pre34[] = {
  { "DINamespace", TF, 5 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 }
};

static const MDTemplate Tmpl_DINamespace_post34[] = {
  { "DINamespace", TF, 5 },
  { "tag",                      DWTagField,    FlgHidden },
  { "file",                     NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "line",                     UnsignedField, 0 }
};

static const MDTemplate Tmpl_DINamespace_5[] = {
  { "DINamespace", TF, 5 },
  { "tag",                      DWTagField,    FlgHidden },
  { "file",                     NodeField,     FlgHidden },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "line",                     UnsignedField, FlgHidden }
};

static const MDTemplate Tmpl_DIModule[] = {
  { "DIModule", TF, 3 },
  { "tag",                      DWTagField,  FlgHidden },
  { "scope",                    NodeField,   0 },
  { "name",                     StringField, 0 }
  //,{ "configMacros",          StringField, FlgOptional },
  //{ "includePath",            StringField, FlgOptional },
  //{ "isysroot",               StringField, FlgOptional }
};

static const MDTemplate Tmpl_DIModule_11[] = {
  { "DIModule", TF, 5 },
  { "tag",                      DWTagField,    FlgHidden },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
};

static const MDTemplate Tmpl_DISubprogram[] = {
  { "DISubprogram", TF, 20 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "isLocal",                  BoolField,         0 },
  { "isDefinition",             BoolField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "isOptimized",              BoolField,         0 },
  { "function",                 ValueField,        0 },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "variables",                NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

static const MDTemplate Tmpl_DISubprogram_13[] = {
  { "DISubprogram", TF, 19 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "spFlags",                  UnsignedField,     0 }, /* TBD: DISPFlag... */
  { "function",                 ValueField,        FlgHidden },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "unit",                     NodeField,         0 },
  { "retainedNodes",            NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

static const MDTemplate Tmpl_DISubprogram_90[] = {
  { "DISubprogram", TF, 18 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "spFlags",                  UnsignedField,     0 }, /* TBD: DISPFlag... */
  { "function",                 ValueField,        FlgHidden },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "unit",                     NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

static const MDTemplate Tmpl_DISubprogram_70[] = {
  { "DISubprogram", TF, 20 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "isLocal",                  BoolField,         0 },
  { "isDefinition",             BoolField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "isOptimized",              BoolField,         0 },
  { "function",                 ValueField,        FlgHidden },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "unit",                     NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

static const MDTemplate Tmpl_DISubprogram_38[] = {
  { "DISubprogram", TF, 20 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "isLocal",                  BoolField,         0 },
  { "isDefinition",             BoolField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "isOptimized",              BoolField,         0 },
  { "function",                 ValueField,        FlgHidden },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "variables",                NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

/** "unit" was added in LLVM 3.9 for DISubprogram */
static const MDTemplate Tmpl_DISubprogram_39[] = {
  { "DISubprogram", TF, 21 },
  { "tag",                      DWTagField,        FlgHidden },
  { "file",                     NodeField,         0 },
  { "scope",                    NodeField,         0 },
  { "name",                     StringField,       0 },
  { "displayName",              StringField,       FlgHidden },
  { "linkageName",              StringField,       0 },
  { "line",                     UnsignedField,     0 },
  { "type",                     NodeField,         0 },
  { "isLocal",                  BoolField,         0 },
  { "isDefinition",             BoolField,         0 },
  { "virtuality",               DWVirtualityField, 0 },
  { "virtualIndex",             UnsignedField,     0 },
  { "containingType",           NodeField,         0 },
  { "flags",                    UnsignedField,     0 }, /* TBD: DIFlag... */
  { "isOptimized",              BoolField,         0 },
  { "function",                 ValueField,        FlgHidden },
  { "templateParams",           NodeField,         0 },
  { "declaration",              NodeField,         0 },
  { "unit",                     NodeField,         0 },
  { "variables",                NodeField,         0 },
  { "scopeLine",                UnsignedField,     0 }
};

static const MDTemplate Tmpl_DILexicalBlock[] = {
  { "DILexicalBlock", TF, 6 },
  { "tag",                      DWTagField,    FlgHidden },
  { "file",                     NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "column",                   UnsignedField, 0 },
  { "discriminator",            UnsignedField, FlgHidden | FlgOptional }
};

static const MDTemplate Tmpl_DILexicalBlock_pre34[] = {
  { "DILexicalBlock", TF, 6 },
  { "tag",                      DWTagField,    FlgHidden },
  { "scope",                    NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "column",                   UnsignedField, 0 },
  { "file",                     NodeField,     0 },
  { "discriminator",            UnsignedField, FlgOptional }
};

static const MDTemplate Tmpl_DILexicalBlockFile[] = {
  { "DILexicalBlock", TF, 4 },
  { "tag",                      DWTagField,    FlgHidden },
  { "file",                     NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "discriminator",            UnsignedField, 0 }
};

static const MDTemplate Tmpl_DIExpression[] = {
  { "DIExpression", TF, 0 }
};

#ifdef FLANG_DEBUGINFO_UNUSED
static const MDTemplate Tmpl_DILocalVariable[] = {
  { "DILocalVariable", TF, 9 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "arg",                      UnsignedField, 0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "type",                     NodeField,     0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "inlinedAt",                UnsignedField, 0 }  /* TBD: NodeField */
};
#endif

static const MDTemplate Tmpl_DILocalVariable_38[] = {
  { "DILocalVariable", TF, 8 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "arg",                      UnsignedField, 0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "type",                     NodeField,     0 },
  { "flags",                    UnsignedField, 0 },/* TBD: DIFlag... */
  { "inlinedAt",                UnsignedField, 0 } /* TBD: NodeField */
};

static const MDTemplate Tmpl_DILocalVariable_embedded_argnum[] = {
  { "DILocalVariable", TF, 8 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line_and_arg",             UnsignedField, 0 },
  { "type",                     NodeField,     0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "inlinedAt",                UnsignedField, 0 }  /* TBD: NodeField */
};

static const MDTemplate Tmpl_DIGlobalVariable[] = {
  { "DIGlobalVariable", TF, 14 },
  { "tag",                      DWTagField,    FlgHidden },
  { "unused",                   NodeField,     FlgHidden },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "displayName",              StringField,   FlgHidden },
  { "linkageName",              StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "type",                     NodeField,     0 },
  { "isLocal",                  BoolField,     0 },
  { "isDefinition",             BoolField,     0 },
  { "variable",                 ValueField,    0 },
  { "flags",                    UnsignedField, 0 },
  { "addrspace",                UnsignedField, FlgOptional } /* nvvm extension */
                                                            /* Missing: declaration */
};

static const MDTemplate Tmpl_DIGlobalVariable4[] = {
  { "DIGlobalVariable", TF, 13 },
  { "tag",                      DWTagField,    FlgHidden },
  { "unused",                   NodeField,     FlgHidden },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "displayName",              StringField,   FlgHidden },
  { "linkageName",              StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "type",                     NodeField,     0 },
  { "isLocal",                  BoolField,     0 },
  { "isDefinition",             BoolField,     0 },
  { "flags",                    UnsignedField, 0 },
  { "addrspace",                UnsignedField, FlgOptional } /* nvvm extension */
};

static const MDTemplate Tmpl_DIGlobalVariableExpression[] = {
  { "DIGlobalVariableExpression", TF, 2 },
  { "var",                      NodeField, 0 },
  { "expr",                     NodeField, 0 }
};

static const MDTemplate Tmpl_DIBasicType_pre34[] = {
  { "DIBasicType", TF, 10 },
  { "tag",                      DWTagField,      0 },
  { "scope",                    NodeField,       0 },
  { "name",                     StringField,     0 },
  { "file",                     NodeField,       0 },
  { "line",                     UnsignedField,   0 },
  { "size",                     UnsignedField,   0 },
  { "align",                    UnsignedField,   0 },
  { "offset",                   UnsignedField,   0 },
  { "flags",                    UnsignedField,   0 }, /* TBD: DIFlag... */
  { "encoding",                 DWEncodingField, 0 }
};

static const MDTemplate Tmpl_DIBasicType[] = {
  { "DIBasicType", TF, 10 },
  { "tag",                      DWTagField,      0 },
  { "unused",                   NodeField,       FlgHidden },
  { "unused",                   NodeField,       FlgHidden },
  { "name",                     StringField,     0 },
  { "line",                     UnsignedField,   0 },
  { "size",                     UnsignedField,   0 },
  { "align",                    UnsignedField,   0 },
  { "offset",                   UnsignedField,   0 },
  { "flags",                    UnsignedField,   0 }, /* TBD: DIFlag... */
  { "encoding",                 DWEncodingField, 0 }
};

/* deprecated */
static const MDTemplate Tmpl_DIStringType_old[] = {
  { "DIBasicType", TF, 5 },
  { "tag",                      DWTagField,      0 },
  { "name",                     StringField,     0 },
  { "size",                     UnsignedField,   0 },
  { "align",                    UnsignedField,   0 },
  { "encoding",                 DWEncodingField, 0 }
};

static const MDTemplate Tmpl_DIStringType[] = {
  { "DIStringType", TF, 7 },
  { "tag",                      DWTagField,    FlgHidden },
  { "name",                     StringField,   0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "encoding",                 UnsignedField, FlgHidden },
  { "stringLength",             NodeField,     0 },
  { "stringLengthExpression",   NodeField,     0 }
};

static const MDTemplate Tmpl_DISubroutineType[] = {
  { "DISubroutineType", TF, 15 },
  { "tag",                      DWTagField,    FlgHidden },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   NodeField,     0 },
  { "name",                     StringField,   0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   NodeField,     0 },
  { "types",                    NodeField,     0 },
  { "unused",                   UnsignedField, 0 },
  { "unused",                   NodeField,     0 },
  { "unused",                   NodeField,     0 },
  { "cc",                       UnsignedField, 0 }
};

static const MDTemplate Tmpl_DIDerivedType_pre34[] = {
  { "DIDerivedType", TF, 10 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "offset",                   UnsignedField, 0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "baseType",                 NodeField,     0 }
};

static const MDTemplate Tmpl_DIDerivedType[] = {
  { "DIDerivedType", TF, 10 },
  { "tag",                      DWTagField,    0 },
  { "file",                     NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "line",                     UnsignedField, 0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "offset",                   UnsignedField, 0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "baseType",                 NodeField,     0 }
};

static const MDTemplate Tmpl_DICompositeType_pre34[] = {
  { "DICompositeType", TF, 13 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "offset",                   UnsignedField, 0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "baseType",                 NodeField,     0 },
  { "elements",                 NodeField,     0 },
  { "runtimeLang",              DWLangField,   0 },
  { "unused",                   NodeField,     FlgHidden }
};

static const MDTemplate Tmpl_DICompositeType[] = {
  { "DICompositeType", TF, 19 },
  { "tag",                      DWTagField,    0 },
  { "file",                     NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "name",                     StringField,   0 },
  { "line",                     UnsignedField, 0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "offset",                   UnsignedField, 0 },
  { "flags",                    UnsignedField, 0 }, /* TBD: DIFlag... */
  { "baseType",                 NodeField,     0 },
  { "elements",                 NodeField,     0 },
  { "runtimeLang",              DWLangField,   0 },
  { "vtableHolder",             NodeField,     0 },
  { "templateParams",           NodeField,     0 },
  { "identifier",               StringField,   0 },
  { "dataLocation",             NodeField,     0 },
  { "associated",               NodeField,     0 },
  { "allocated",                NodeField,     0 },
  { "rank",                     NodeField,     0 }
};

static const MDTemplate Tmpl_DIFortranArrayType[] = {
  { "DIFortranArrayType", TF, 7 },
  { "tag",                      DWTagField,    0 },
  { "scope",                    NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "size",                     UnsignedField, 0 },
  { "align",                    UnsignedField, 0 },
  { "baseType",                 NodeField,     0 },
  { "elements",                 NodeField,     0 }
};

static const MDTemplate Tmpl_DISubrange_pre11[] = {
  { "DISubrange", TF, 3 },
  { "tag",                      DWTagField,  FlgHidden },
  { "lowerBound",               SignedField, 0 },
  { "count",                    SignedField, FlgMandatory }
};

static const MDTemplate Tmpl_DISubrange[] = {
  { "DISubrange", TF, 5 },
  { "tag",                      DWTagField,      FlgHidden },
  { "count",                    SignedOrMDField, 0 },
  { "lowerBound",               SignedOrMDField, 0 },
  { "upperBound",               SignedOrMDField, 0 },
  { "stride",                   SignedOrMDField, 0 }
};

static const MDTemplate Tmpl_DIGenericSubrange[] = {
  { "DIGenericSubrange", TF, 4 },
  { "tag",                      DWTagField,      FlgHidden },
  { "lowerBound",               SignedOrMDField, 0 },
  { "upperBound",               SignedOrMDField, FlgMandatory },
  { "stride",                   SignedOrMDField, FlgMandatory }
};

static const MDTemplate Tmpl_DISubrange_pre37[] = {
  { "DISubrange", TF, 3 },
  { "tag",                      DWTagField,  FlgHidden },
  { "lowerBound",               SignedField, 0 },
  { "upperBound",               SignedField, 0 }
};

static const MDTemplate Tmpl_DIFortranSubrange[] = {
  { "DIFortranSubrange", TF, 7 },
  { "tag",                      DWTagField,  FlgHidden },
  { "constLowerBound",          SignedField, 0 },
  { "constUpperBound",          SignedField, FlgSkip1 },
  { "lowerBound",               NodeField,   0 },
  { "lowerBoundExpression",     NodeField,   0 },
  { "upperBound",               NodeField,   0 },
  { "upperBoundExpression",     NodeField,   0 }
};

static const MDTemplate Tmpl_DIEnumerator[] = {
  { "DIEnumerator", TF, 3 },
  { "tag",                      DWTagField,  FlgHidden },
  { "name",                     StringField, 0 },
  { "value",                    SignedField, FlgMandatory }
};

static const MDTemplate Tmpl_DIImportedEntity_pre11[] = {
  { "DIImportedEntity", TF, 6 },
  { "tag",                      DWTagField,    0 },
  { "entity",                   NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "name",                     StringField,   0 }
};

static const MDTemplate Tmpl_DIImportedEntity[] = {
  { "DIImportedEntity", TF, 7 },
  { "tag",                      DWTagField,    0 },
  { "entity",                   NodeField,     0 },
  { "scope",                    NodeField,     0 },
  { "file",                     NodeField,     0 },
  { "line",                     UnsignedField, 0 },
  { "name",                     StringField,   0 },
  { "elements",                 NodeField,     0 }
};

static const MDTemplate Tmpl_DICommonBlock[] = {
  { "DICommonBlock", TF, 3 },
  { "scope",                    NodeField,   0 },
  { "declaration",              NodeField,   0 },
  { "name",                     StringField, 0 }
};

/* clang-format on */

#undef TF

/**
   \brief Write out an \ref LL_MDRef from \p module.
   \param out   output file
   \param module  the LLVM module
   \param rmdref  the metadata node to be written
   \param omit_metadata_type  if true then omit \c metadata keyword

   This functions writes a metadata reference as it appears in metadata context,
   such as inside a metadata node definition, or following a \c !dbg tag.

   Metadata references may use a different syntax when used as function
   arguments, depending on the LLVM version. That is \e not dealt with by this
   function.
 */
void
write_mdref(FILE *out, LL_Module *module, LL_MDRef rmdref,
            int omit_metadata_type)
{
  const char *tag = "metadata ";
  LL_MDRef mdref = rmdref;

  /* The metadata type tag is omitted in metadata context in LLVM 3.6+, and
   * always in named metadata definitions. */
  if (omit_metadata_type)
    tag = "";

  switch (LL_MDREF_kind(mdref)) {
  case MDRef_Node:
    if (LL_MDREF_value(mdref))
      fprintf(out, "%s!%u", tag, LL_MDREF_value(mdref));
    else
      fprintf(out, "null");
    break;

  case MDRef_String:
    assert(LL_MDREF_value(mdref) < module->mdstrings_count, "Bad string MDRef",
           LL_MDREF_value(mdref), ERR_Fatal);
    fprintf(out, "%s%s", tag, module->mdstrings[LL_MDREF_value(mdref)]);
    break;

  case MDRef_Constant:
    assert(LL_MDREF_value(mdref) < module->constants_count,
           "Bad constant MDRef", LL_MDREF_value(mdref), ERR_Fatal);
    fprintf(out, "%s %s",
            module->constants[LL_MDREF_value(mdref)]->type_struct->str,
            module->constants[LL_MDREF_value(mdref)]->data);
    break;

  case MDRef_SmallInt1:
    fprintf(out, "i1 %u", LL_MDREF_value(mdref));
    break;

  case MDRef_SmallInt32:
    fprintf(out, "i32 %u", LL_MDREF_value(mdref));
    break;

  case MDRef_SmallInt64:
    fprintf(out, "i64 %u", LL_MDREF_value(mdref));
    break;

  default:
    interr("Invalid MDRef kind", LL_MDREF_kind(mdref), ERR_Fatal);
  }
}

/**
   \brief generate full DWARF debug emission mode
 */
static const char *
dwarf_emission_name(int value)
{
  switch (value) {
  case 2:
    return "NoDebug";
  case 3:
    return "LineTablesOnly";
  default:
    return "FullDebug";
  }
}

/**
   \brief generate DWARF table kind
 */
static const char *
dwarf_table_name(int value)
{
  switch (value) {
  case 0:
    return "Default";
  case 1:
    return "GNU";
  case 2:
    return "None";
  default:
    return "None";
  }
}

/**
   \brief Write out an an LL_MDRef as a field in a specialised MDNode class
   \param out        file to write to
   \param module       module containing the metadata
   \param node         the metadata node to be written
   \param needs_comma  If true, print a ", " before the field label
   \return true iff the field was actually printed

   Includes priting the "name:" label.  The field is not printed if it has its
   default value.

   The formatting is guided by the field type from the MDTemplate, and the
   MDRef types are validated.
 */
static int
write_mdfield(FILE *out, LL_Module *module, int needs_comma, LL_MDRef mdref,
              const MDTemplate *tmpl)
{
  unsigned value = LL_MDREF_value(mdref);
  const char *prefix = needs_comma ? ", " : "";
  const bool mandatory = (tmpl->flags & FlgMandatory) != 0;

  if (tmpl->flags & FlgHidden)
    return false;

  switch (LL_MDREF_kind(mdref)) {
  case MDRef_Node:
    if (value) {
      assert(tmpl->type == NodeField || tmpl->type == SignedOrMDField,
             "metadata elem should not be a mdnode", tmpl->type, ERR_Fatal);
      fprintf(out, "%s%s: !%u", prefix, tmpl->name, value);
    } else if (mandatory) {
      fprintf(out, "%s%s: null", prefix, tmpl->name);
    } else {
      return false;
    }
    break;

  case MDRef_String:
    assert(tmpl->type == StringField, "metadata elem should not be a string",
           tmpl->type, ERR_Fatal);
    assert(value < module->mdstrings_count, "Bad string MDRef", value,
           ERR_Fatal);
    if (!mandatory && strcmp(module->mdstrings[value], "!\"\"") == 0)
      return false;
    /* The mdstrings[] entry is formatted as !"...". Strip the leading !. */
    fprintf(out, "%s%s: %s", prefix, tmpl->name, module->mdstrings[value] + 1);
    break;

  case MDRef_Constant:
    assert(value < module->constants_count, "Bad constant MDRef", value,
           ERR_Fatal);
    switch (tmpl->type) {
    case ValueField:
      fprintf(out, "%s%s: %s %s", prefix, tmpl->name,
              module->constants[value]->type_struct->str,
              module->constants[value]->data);
      break;

#ifdef HOST_WIN
#define strtoll _strtoi64
#endif
    case UnsignedField:
      if (module->constants[value]->data[0] == '-') {
        /* The value stored is negative.  LLVM expects it to be unsigned, so
           convert it to be positive. */
        long long intval = strtoll(module->constants[value]->data, NULL, 10);
        if ((long long)INT_MIN <= intval && intval < 0) {
          /* It was most likely a 32 bit value originally. */
          fprintf(out, "%s%s: %u", prefix, tmpl->name, (unsigned)(int)intval);
        } else {
          fprintf(out, "%s%s: %llu", prefix, tmpl->name, intval);
        }
      } else {
        fprintf(out, "%s%s: %s", prefix, tmpl->name,
                module->constants[value]->data);
      }
      break;

    case SignedOrMDField:
    case SignedField: {
      bool doOutput = true;
      const char *dv = module->constants[value]->data;
      if (tmpl->flags & FlgSkip1) {
        const ISZ_T M = 1ul << ((sizeof(ISZ_T) * 8) - 1);
        ISZ_T idv;
        sscanf(dv, "%" ISZ_PF "d", &idv);
        doOutput = (idv != M);
      }
      if (!doOutput)
        return false;
      fprintf(out, "%s%s: %s", prefix, tmpl->name, dv);
    } break;

    default:
      interr("metadata elem should not be a value", tmpl->type, ERR_unused);
    }
    break;

  case MDRef_SmallInt1:
  case MDRef_SmallInt32:
  case MDRef_SmallInt64:
    if (!value && !mandatory)
      return false;
    switch (tmpl->type) {
    case UnsignedField:
    case SignedField:
    case SignedOrMDField:
      fprintf(out, "%s%s: %u", prefix, tmpl->name, value);
      break;

    case BoolField:
      assert(value <= 1, "boolean value expected", value, ERR_Fatal);
      fprintf(out, "%s%s: %s", prefix, tmpl->name, value ? "true" : "false");
      break;

    case DWTagField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name,
              dwarf_tag_name(value & 0xffff));
      break;

    case DWLangField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name, dwarf_lang_name(value));
      break;

    case DWVirtualityField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name,
              dwarf_virtuality_name(value));
      break;

    case DWEncodingField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name, dwarf_encoding_name(value));
      break;

    case DWEmissionField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name, dwarf_emission_name(value));
      break;

    case DebugNameTableKindField:
      fprintf(out, "%s%s: %s", prefix, tmpl->name, dwarf_table_name(value));
      break;

    default:
      interr("metadata elem should not be an int", tmpl->type, ERR_unused);
    }
    break;

  default:
    interr("Invalid MDRef kind", LL_MDREF_kind(mdref), ERR_Fatal);
  }

  return true;
}

/*
 * Write out a metadata node definition in the "plain" style: !{ !1, ... }.
 *
 * When omit_metadata_type is set, don't print out the leading "metadata" type
 * tag. This doesn't affect the printing of the internal mdnode contents.
 */
static void
write_mdnode_plain(FILE *out, LL_Module *module, const LL_MDNode *node,
                   int omit_metadata_type)
{
  unsigned i;

  if (!omit_metadata_type)
    fprintf(out, "metadata ");

  if (ll_feature_use_distinct_metadata(&module->ir) && node->is_distinct)
    fprintf(out, "distinct ");

  fprintf(out, "!{ ");
  for (i = 0; i < node->num_elems; i++) {
    LL_MDRef mdref = LL_MDREF_INITIALIZER(0, 0);
    mdref = node->elem[i];
    if (i > 0)
      fprintf(out, ", ");
    write_mdref(out, module, mdref, omit_metadata_type);
  }
  fprintf(out, " }\n");
}

/*
 * Write out a metadata node in the specialized form: !MDLocation(line: 42,
 * ...).
 *
 * Also perform some basic schema validation against the provided template.
 */
static void
write_mdnode_spec(FILE *out, LL_Module *module, const LL_MDNode *node,
                  const MDTemplate *tmpl)
{
  const unsigned num_fields = tmpl->flags;
  unsigned i;
  int needs_comma = false;

  if (ll_feature_use_distinct_metadata(&module->ir) && node->is_distinct)
    fprintf(out, "distinct ");

  assert(node->num_elems <= num_fields, "metadata node has too many fields.",
         node->num_elems, ERR_Fatal);

  fprintf(out, "!%s(", tmpl->name);
  for (i = 0; i < node->num_elems; i++)
    if (write_mdfield(out, module, needs_comma, node->elem[i], &tmpl[i + 1]))
      needs_comma = true;
  fprintf(out, ")\n");
}

/**
   \brief Get the textual name for module-level named metadata.
 */
static const char *
get_metadata_name(LL_MDName name)
{
  switch (name) {
  case MD_llvm_module_flags:
    return "!llvm.module.flags";
  case MD_llvm_dbg_cu:
    return "!llvm.dbg.cu";
  case MD_llvm_linker_options:
    return "!llvm.linker.options";
  case MD_opencl_kernels:
    return "!opencl.kernels";
  case MD_nvvm_annotations:
    return "!nvvm.annotations";
  case MD_nvvmir_version:
    return "!nvvmir.version";
  default:
    interr("Unknown metadata name", name, ERR_Fatal);
  }
  return NULL;
}

typedef const LL_MDNode *MDNodeRef;

static void emitRegular(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDICompileUnit(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIFile(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIBasicType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIBasicStringType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIStringType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDISubroutineType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIDerivedType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDICompositeType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIFortranArrayType(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDISubRange(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIFortranSubrange(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIEnumerator(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDINamespace(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIModule(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIGlobalVariable(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDISubprogram(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDILexicalBlock(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDILexicalBlockFile(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDILocation(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDILocalVariable(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIExpression(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIGlobalVariableExpression(FILE *, LLVMModuleRef, MDNodeRef,
                                           unsigned);
static void emitDIImportedEntity(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDICommonBlock(FILE *, LLVMModuleRef, MDNodeRef, unsigned);
static void emitDIGenericSubRange(FILE *, LLVMModuleRef, MDNodeRef, unsigned);

typedef void (*MDDispatchMethod)(FILE *out, LLVMModuleRef mod, MDNodeRef mdnode,
                                 unsigned mdi);

typedef struct MDDispatch {
  MDDispatchMethod method;
} MDDispatch;

static MDDispatch mdDispTable[LL_MDClass_MAX] = {
    {emitRegular},                    // LL_PlainMDNode
    {emitDICompileUnit},              // LL_DICompileUnit
    {emitDIFile},                     // LL_DIFile
    {emitDIBasicType},                // LL_DIBasicType
    {emitDISubroutineType},           // LL_DISubroutineType
    {emitDIDerivedType},              // LL_DIDerivedType
    {emitDICompositeType},            // LL_DICompositeType
    {emitDIFortranArrayType},         // LL_DIFortranArrayType
    {emitDISubRange},                 // LL_DISubRange
    {emitDIFortranSubrange},          // LL_DIFortranSubrange
    {emitDIEnumerator},               // LL_DIEnumerator
    {emitRegular},                    // LL_DITemplateTypeParameter
    {emitRegular},                    // LL_DITemplateValueParameter
    {emitDINamespace},                // LL_DINamespace
    {emitDIModule},                   // LL_DIModule
    {emitDIGlobalVariable},           // LL_DIGlobalVariable
    {emitDISubprogram},               // LL_DISubprogram
    {emitDILexicalBlock},             // LL_DILexicalBlock
    {emitDILexicalBlockFile},         // LL_DILexicalBlockFile
    {emitDILocation},                 // LL_DILocation
    {emitDILocalVariable},            // LL_DILocalVariable
    {emitDIExpression},               // LL_DIExpression
    {emitRegular},                    // LL_DIObjCProperty
    {emitDIImportedEntity},           // LL_DIImportedEntity
    {emitDIGlobalVariableExpression}, // LL_DIGlobalVariableExpression
    {emitDIBasicStringType},          // LL_DIBasicType_string - deprecated
    {emitDIStringType},               // LL_DIStringType
    {emitDICommonBlock},              // LL_DICommonBlock
    {emitDIGenericSubRange},          // LL_DIGenericSubRange
};

INLINE static void
emitRegularPrefix(FILE *out, unsigned mdi)
{
  fprintf(out, "!%u = ", mdi);
}

/** Simple helper function */
INLINE static bool
useSpecialized(LLVMModuleRef mod)
{
  return ll_feature_use_specialized_mdnodes(&mod->ir);
}

INLINE static void
emitRegular(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode, unsigned mdi)
{
  emitRegularPrefix(out, mdi);
  write_mdnode_plain(out, mod, mdnode, ll_feature_omit_metadata_type(&mod->ir));
}

INLINE static void
emitSpec(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode, unsigned mdi,
         const MDTemplate *tmpl)
{
  emitRegularPrefix(out, mdi);
  write_mdnode_spec(out, mod, mdnode, tmpl);
}

INLINE static void
emitUnspec(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode, unsigned mdi,
           const MDTemplate *tmpl)
{
  fputs("; ", out);
  emitSpec(out, mod, mdnode, mdi, tmpl);
  emitRegular(out, mod, mdnode, mdi);
}

static void
emitTmpl(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode, unsigned mdi,
         const MDTemplate *tmpl)
{
  if (useSpecialized(mod)) {
    emitSpec(out, mod, mdnode, mdi, tmpl);
    return;
  }
  emitUnspec(out, mod, mdnode, mdi, tmpl);
}

static void
emitDICompileUnit(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                  unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DICompileUnit_pre34);
    return;
  }
  if (ll_feature_subprogram_not_in_cu(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DICompileUnit_ver39);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DICompileUnit);
}

static void
emitDIFile(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode, unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIFile_pre34);
    return;
  }
  if (LL_MDREF_kind(mdnode->elem[0]) == MDRef_String) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIFile_pair);
    return;
  }
  emitUnspec(out, mod, mdnode, mdi, Tmpl_DIFile_tagged);
}

static void
emitDIBasicType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIBasicType_pre34);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIBasicType);
}

static void
emitDIStringType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                 unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIStringType);
}

/* deprecated */
static void
emitDIBasicStringType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                      unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIStringType_old);
}

static void
emitDISubroutineType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                     unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubroutineType);
}

static void
emitDIDerivedType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                  unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIDerivedType_pre34);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIDerivedType);
}

static void
emitDICompositeType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                    unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DICompositeType_pre34);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DICompositeType);
}

static void
emitDIFortranArrayType(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                       unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIFortranArrayType);
}

static void
emitDISubRange(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
               unsigned mdi)
{
  if (ll_feature_debug_info_ver90(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubrange);
    return;
  }
  if (!ll_feature_debug_info_subrange_needs_count(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubrange_pre37);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubrange_pre11);
}

static void
emitDIGenericSubRange(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                      unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIGenericSubrange);
}

static void
emitDIFortranSubrange(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                      unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIFortranSubrange);
}

static void
emitDIEnumerator(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                 unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIEnumerator);
}

static void
emitDINamespace(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DINamespace_pre34);
    return;
  }
  if (ll_feature_no_file_in_namespace(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DINamespace_5);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DINamespace_post34);
}

static void
emitDIModule(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnd, unsigned mdi)
{
  if (ll_feature_debug_info_ver11(&mod->ir)) {
    emitTmpl(out, mod, mdnd, mdi, Tmpl_DIModule_11);
    return;
  }
  emitTmpl(out, mod, mdnd, mdi, Tmpl_DIModule);
}

static void
emitDIGlobalVariable(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                     unsigned mdi)
{
  if (ll_feature_from_global_to_md(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIGlobalVariable4);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIGlobalVariable);
}

static void
emitDISubprogram(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                 unsigned mdi)
{
  if (!ll_feature_debug_info_pre34(&mod->ir)) {
    if (ll_feature_debug_info_ver13(&mod->ir)) {
      // 13.0, 'retainedNodes:' is back
      emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram_13);
      return;
    }
    if (ll_feature_debug_info_ver90(&mod->ir)) {
      // 9.0, 'isLocal:', 'isDefinition:', and 'isOptimized:' removed
      // and 'spFlags:' added
      emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram_90);
      return;
    }
    if (ll_feature_debug_info_ver70(&mod->ir)) {
      // 7.0, 'variables:' was removed
      emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram_70);
      return;
    }
    if (ll_feature_subprogram_not_in_cu(&mod->ir)) {
      // 3.9, 'unit:' was added
      emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram_39);
      return;
    }
    if (ll_feature_debug_info_ver38(&mod->ir)) {
      // 3.8, 'function:' was removed
      emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram_38);
      return;
    }
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DISubprogram);
    return;
  }
  emitRegular(out, mod, mdnode, mdi);
}

static void
emitDILexicalBlock(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                   unsigned mdi)
{
  if (ll_feature_debug_info_pre34(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DILexicalBlock_pre34);
    return;
  }
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DILexicalBlock);
}

static void
emitDILexicalBlockFile(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                       unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DILexicalBlockFile);
}

static void
emitDILocation(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
               unsigned mdi)
{
  const MDTemplate *tmpl = ll_feature_debug_info_DI_syntax(&mod->ir)
                               ? Tmpl_DILocation
                               : Tmpl_MDLocation;
  if (ll_feature_debug_info_mdlocation(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, tmpl);
    return;
  }
  emitUnspec(out, mod, mdnode, mdi, tmpl);
}

static void
emitDILocalVariable(FILE *out, LLVMModuleRef mod, const LL_MDNode *node,
                    unsigned mdi)
{
  if (ll_feature_dbg_local_variable_embeds_argnum(&mod->ir)) {
    emitTmpl(out, mod, node, mdi, Tmpl_DILocalVariable_embedded_argnum);
    return;
  }
  if (ll_feature_debug_info_ver38(&mod->ir)) {
    // 3.8, 'tag:' was removed
    emitTmpl(out, mod, node, mdi, Tmpl_DILocalVariable_38);
    return;
  }
  emitTmpl(out, mod, node, mdi, Tmpl_DILocation);
}

static void
emitDIImportedEntity(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                     unsigned mdi)
{
  if (ll_feature_debug_info_ver11(&mod->ir)) {
    emitTmpl(out, mod, mdnode, mdi, Tmpl_DIImportedEntity);
    return;
  }

  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIImportedEntity_pre11);
}

static void
emitDICommonBlock(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                  unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DICommonBlock);
}

INLINE static const char *
ll_dw_op_to_name(LL_DW_OP_t op)
{
  switch (op) {
  case LL_DW_OP_deref:
    return "DW_OP_deref";
  case LL_DW_OP_plus:
    return "DW_OP_plus";
  case LL_DW_OP_minus:
    return "DW_OP_minus";
  case LL_DW_OP_dup:
    return "DW_OP_dup";
  case LL_DW_OP_LLVM_fragment:
    return "DW_OP_LLVM_fragment";
  case LL_DW_OP_swap:
    return "DW_OP_swap";
  case LL_DW_OP_xderef:
    return "DW_OP_xderef";
  case LL_DW_OP_stack_value:
    return "DW_OP_stack_value";
  case LL_DW_OP_constu:
    return "DW_OP_constu";
  case LL_DW_OP_plus_uconst:
    return "DW_OP_plus_uconst";
  case LL_DW_OP_push_object_address:
    return "DW_OP_push_object_address";
  case LL_DW_OP_mul:
    return "DW_OP_mul";
  case LL_DW_OP_over:
    return "DW_OP_over";
  case LL_DW_OP_and:
    return "DW_OP_and";
  default:
    break;
  }
  DEBUG_ASSERT(false, "unhandled LL_DW_OP_t");
  return "*bug*";
}

INLINE static const char *
decode_expression_op(LLVMModuleRef mod, LL_MDRef md, char *buff)
{
  int value;
  bool isLiteralOp;

  if (LL_MDREF_kind(md) == MDRef_Constant) {
    strcpy(buff, mod->constants[LL_MDREF_value(md)]->data);
    return buff;
  }
  DEBUG_ASSERT(LL_MDREF_kind(md) == MDRef_SmallInt32, "not int");
  value = LL_MDREF_value(md);
  isLiteralOp = value & 1;
  value >>= 1;
  if (isLiteralOp)
    return ll_dw_op_to_name((LL_DW_OP_t)value);
  sprintf(buff, "%d", value);
  return buff;
}

static void
emitComplexDIExpression(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                        unsigned mdi)
{
  unsigned i;
  unsigned cnt = mdnode->num_elems;
  char buff[32];

  emitRegularPrefix(out, mdi);
  fputs("!DIExpression(", out);
  for (i = 0; i < cnt; ++i) {
    if (i > 0)
      fputs(", ", out);
    fputs(decode_expression_op(mod, mdnode->elem[i], buff), out);
  }
  fputs(")\n", out);
}

static void
emitDIExpression(FILE *out, LLVMModuleRef mod, const LL_MDNode *mdnode,
                 unsigned mdi)
{
  if (useSpecialized(mod)) {
    if (mdnode->num_elems > 0) {
      emitComplexDIExpression(out, mod, mdnode, mdi);
      return;
    }
    emitSpec(out, mod, mdnode, mdi, Tmpl_DIExpression);
    return;
  }
  if (mdnode->num_elems > 0) {
    fputs("; ", out);
    emitComplexDIExpression(out, mod, mdnode, mdi);
  }
  emitUnspec(out, mod, mdnode, mdi, Tmpl_DIExpression);
}

static void
emitDIGlobalVariableExpression(FILE *out, LLVMModuleRef mod,
                               const LL_MDNode *mdnode, unsigned mdi)
{
  emitTmpl(out, mod, mdnode, mdi, Tmpl_DIGlobalVariableExpression);
}

static void
write_metadata_node(FILE *out, LLVMModuleRef module, MDNodeRef node,
                    unsigned mdi)
{
  const LL_MDClass mdClass = node->mdclass;
  DEBUG_ASSERT(mdClass < LL_MDClass_MAX, "mdclass out of bounds");
  mdDispTable[mdClass].method(out, module, node, mdi);
}

#ifdef __cplusplus
inline LL_MDName
NextMDName(LL_MDName &name)
{
  name = static_cast<LL_MDName>(static_cast<unsigned>(name) + 1);
  return name;
}
#else
#define NextMDName(N) ++(N)
#endif

/**
   \brief Write out all the module metadata

   Write out all the so-called named metadata and then the regular metadata
 */
void
ll_write_metadata(FILE *out, LLVMModuleRef module)
{
  fprintf(out, "\n; Named metadata\n");
  for (LL_MDName i = MD_llvm_module_flags; i < MD_NUM_NAMES; NextMDName(i)) {
    const LL_MDNode *node = module->named_mdnodes[i];
    if (node) {
      fprintf(out, "%s = ", get_metadata_name(i));
      write_mdnode_plain(out, module, node, /* omit_metadata_type = */ true);
    }
  }

  fprintf(out, "\n; Metadata\n");
  for (unsigned j = 0; j < module->mdnodes_count; j++) {
    write_metadata_node(out, module, module->mdnodes[j], j + 1);
  }
}

void
ll_write_global_var_signature(FILE *out, LL_Value *variable)
{
  if (variable->mvtype == LL_GLOBAL) {
    fprintf(out, "global ptr");
  } else {
    fprintf(out, "ptr");
  }
  fprintf(out, " %s", variable->data);
}

/**
   \brief Write definition of the special <code>\@llvm.used</code> global
 */
void
ll_write_llvm_used(FILE *out, LLVMModuleRef module)
{
  unsigned i;

  if (!module->num_llvm_used)
    return;

  fprintf(out, "@llvm.used = appending global [%u x ptr] [\n  ",
          module->num_llvm_used);
  for (i = 0; i < module->num_llvm_used; i++) {
    LL_Value *ptr = module->llvm_used.values[i];
    if (i)
      fprintf(out, ",\n  ");
    fprintf(out, "%s %s", ptr->type_struct->str, ptr->data);
  }
  fprintf(out, "\n], section \"llvm.metadata\"\n");
}
#ifdef OMP_OFFLOAD_LLVM
void ll_build_metadata_device(FILE *out, LLVMModuleRef module)
{
  LL_Function *function;
  /* Create kernel descriptors. */
  for (function = module->first; function; function = function->next) {
    LLMD_Builder mdb;

    if (!function->is_kernel)
    continue;

    mdb = llmd_init(module);
    llmd_add_value(mdb, ll_get_function_pointer(module, function));
    llmd_add_string(mdb, "kernel");
    llmd_add_i32(mdb, 1);
    ll_extend_named_md_node(module, MD_nvvm_annotations, llmd_finish(mdb));

    mdb = llmd_init(module);
    llmd_add_value(mdb, ll_get_function_pointer(module, function));
    if (function->launch_bounds > 0) {
      llmd_add_string(mdb, "maxntidx");
      llmd_add_i32(mdb, function->launch_bounds);
      llmd_add_string(mdb, "maxntidy");
      llmd_add_i32(mdb, 1);
      llmd_add_string(mdb, "maxntidz");
      llmd_add_i32(mdb, 1);
    }
    //dunno whether I need it or not at the moment
    //ll_extend_named_md_node(module, MD_nvvm_annotations, llmd_finish(mdb));
  }
}
#endif
/**
   \brief Write out definitions or declarations of global LL_Objects.

   If this function is called more than once, only the new objects added since
   the last call will be written.
 */
void
ll_write_global_objects(FILE *out, LLVMModuleRef module)
{
  LL_Object *object;

  for (object = module->first_global; object; object = object->next) {
    int addrspace = ll_get_pointer_addrspace(object->address.type_struct);

    fprintf(out, "%s =", object->address.data);

    /* TBD: [Linkage] [Visibility] [DLLStorageClass] [ThreadLocal]
     * [unnamed_addr] */

    if (addrspace && object->kind != LLObj_Alias)
      fprintf(out, " addrspace(%d)", addrspace);

    /* Linkage */
    if (object->linkage != LL_EXTERNAL_LINKAGE)
      fprintf(out, " %s", ll_get_linkage_string(object->linkage));

    /* Kind */
    switch (object->kind) {
    case LLObj_Global:
      fprintf(out, " global ");
      break;
    case LLObj_Const:
      fprintf(out, " constant ");
      break;
    case LLObj_Alias:
      fprintf(out, " alias ");
      break;
    default:
      interr("ll_write_global_objects: invalid global kind", object->kind,
             ERR_Fatal);
    }

    /* Print an initializer following the type. */
    switch (object->init_style) {
    case LLInit_Declaration:
      fprintf(out, "%s", object->type->str);
      break;
    case LLInit_Zero:
      fprintf(out, "%s zeroinitializer", object->type->str);
      break;
    case LLInit_ConstExpr:
      fprintf(out, "%s %s", object->init_data.const_expr->type_struct->str,
              object->init_data.const_expr->data);
      break;
    case LLInit_Function:
      /* Call the provided function pointer which will print out the
       * initializer with the leading type. */
      object->init_data.function(out, object);
      break;
    }

    /* Alignment */
    if (object->align_bytes)
      fprintf(out, ", align %u", object->align_bytes);

    /* TBD: [, section "name"] [, comdat ...] */
    fprintf(out, "\n");
  }

  /* Reset the list of global objects so this function can be called multiple
   * times without creating duplicates. */
  module->first_global = NULL;
  module->last_global = NULL;
}

void
ll_write_module(FILE *out, LL_Module *module, int generate_no_return_variants, const char *no_return_prefix)
{
  clear_prototypes();
  ll_write_module_header(out, module);

  fprintf(out, "; Begin User structs\n");
  ll_write_user_structs(out, module);
  fprintf(out, "; End User structs\n\n");

  fprintf(out, "; Begin module variables\n");
  /* HACKERY */
  for (unsigned i = 0; i < module->num_module_vars; i++) {
    const char *linkage_string;
    int addrspace;
    const char *initializer;

    switch (module->module_vars.values[i]->linkage) {
    case LL_EXTERNAL_LINKAGE:
      initializer = "";
      break;
    case LL_COMMON_LINKAGE:
      initializer = "zeroinitializer";
      break;
    case LL_INTERNAL_LINKAGE:
      initializer = "zeroinitializer";
      break;
    case LL_NO_LINKAGE:
      initializer = "zeroinitializer";
      break;
    case LL_WEAK_LINKAGE:
      /* ICE */
      initializer = "";
      break;
    }
    linkage_string =
        ll_get_linkage_string(module->module_vars.values[i]->linkage);

    if (module->module_vars.values[i]->mvtype == LL_GLOBAL) {
      fprintf(out, "%s = external addrspace(%d) global [0 x double]\n",
              module->module_vars.values[i]->data,
              module->module_vars.values[i]->type_struct->addrspace);
    } else if (module->module_vars.values[i]->mvtype == LL_DEVICE) {
      unsigned int align_val;

      align_val = module->module_vars.values[i]->align_bytes;
      if (align_val == 0) {
        /* Enforce alignment to 16-bytes, if no alignment specified */
        align_val = 16;
      }
      fprintf(out, "%s = %s addrspace(1) global %s %s, align %u\n",
              module->module_vars.values[i]->data, linkage_string,
              module->module_vars.values[i]->type_struct->str, initializer,
              align_val);
    } else if (module->module_vars.values[i]->mvtype == LL_CONSTANT) {
      fprintf(out, "%s = %s addrspace(4) global %s %s, align 16\n",
              module->module_vars.values[i]->data, linkage_string,
              module->module_vars.values[i]->type_struct->str, initializer);
    } else if (module->module_vars.values[i]->linkage == LL_EXTERNAL_LINKAGE) {
      fprintf(out, "%s = %s addrspace(%d) global %s\n",
              module->module_vars.values[i]->data, linkage_string,
              (module->module_vars.values[i]->storage
                   ? module->module_vars.values[i]
                         ->storage->type_struct->sub_types[0]
                         ->addrspace
                   : 0),
              module->module_vars.values[i]->type_struct->str);
    } else {
      char align_str[80];
      switch (module->module_vars.values[i]->type_struct->data_type) {
      case LL_I1:
      case LL_I8:
      case LL_I16:
      case LL_I32:
      case LL_I64:
      case LL_FLOAT:
      case LL_DOUBLE:
      case LL_PTR:
        if (module->module_vars.values[i]->flags & VAL_IS_TEXTURE)
          linkage_string = "";
        break;
      default:
        break;
      }
      addrspace = 0;
      if (module->module_vars.values[i]->storage) {
        addrspace = module->module_vars.values[i]
                        ->storage->type_struct->sub_types[0]
                        ->addrspace;
      }
      align_str[0] = '\0';
      if (module->module_vars.values[i]->align_bytes)
        sprintf(align_str, ", align %d",
                module->module_vars.values[i]->align_bytes);
      fprintf(out, "%s = %s addrspace(%d) global %s %s%s\n",
              module->module_vars.values[i]->data, linkage_string, addrspace,
              module->module_vars.values[i]->type_struct->str, initializer,
              align_str);
    }
  }
  ll_write_global_objects(out, module);
  /* TODO: This needs to be enabled generally */
  ll_write_llvm_used(out, module);
  fprintf(out, "; End module variables\n\n");

  if (generate_no_return_variants) {
    fprintf(out, "declare void @llvm.nvvm.exit() noreturn\n");
  }

  int num_functions = 0;
  LL_Function *function = module->first;
  while (function) {
    ll_write_function(out, function, module, false, "");
    if (generate_no_return_variants) {
      ll_write_function(out, function, module, true, no_return_prefix);
    }
    function = function->next;
    num_functions++;
  }
  write_prototypes(out, module);
  ll_write_metadata(out, module);
}
