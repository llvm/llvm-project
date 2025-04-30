/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LL_WRITE_H_
#define LL_WRITE_H_

#include <stdio.h>
#include "ll_structure.h"

/**
   \brief ...
 */
void ll_write_basicblock(FILE *out, LL_Function *function, LL_BasicBlock *block,
                         LL_Module *module, int no_return);

/**
   \brief ...
 */
void ll_write_function(FILE *out, LL_Function *function, LL_Module *module, 
                       bool no_return, const char *prefix);

/**
   \brief ...
 */
void ll_write_global_objects(FILE *out, LLVMModuleRef module);

/**
   \brief ...
 */
void ll_write_global_var_signature(FILE *out, LL_Value *variable);

/**
   \brief ...
 */
void ll_write_instruction(FILE *out, struct LL_Instruction_ *inst,
                          LL_Module *module, int no_return);

/**
   \brief ...
 */
void ll_write_llvm_used(FILE *out, LLVMModuleRef module);

/**
   \brief ...
 */
void ll_write_local_objects(FILE *out, LL_Function *function);

/**
   \brief ...
 */
void ll_write_metadata(FILE *out, LLVMModuleRef module);

/**
   \brief ...
 */
void ll_write_module(FILE *out, LL_Module *module, int no_return,
                     const char *no_return_prefix);

/**
   \brief ...
 */
void ll_write_module_header(FILE *out, LLVMModuleRef module);

/**
   \brief ...
 */
void ll_write_object_dbg_references(FILE *out, LL_Module *m,
                                    LL_ObjToDbgList *ods);

/**
   \brief ...
 */
void ll_write_user_structs(FILE *out, LLVMModuleRef module);

/**
   \brief ...
 */
void write_mdref(FILE *out, LL_Module *module, LL_MDRef rmdref,
                 int omit_metadata_type);

/**
   \brief Build llvm metadata information specificly for llvm nvptx backend.
 */
void ll_build_metadata_device(FILE *out, LLVMModuleRef module);

#endif
