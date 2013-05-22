//===-- ClangExpressionDeclMap.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionDeclMap.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Decl.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

ClangExpressionDeclMap::ClangExpressionDeclMap (bool keep_result_in_memory, ExecutionContext &exe_ctx) :
    ClangASTSource (exe_ctx.GetTargetSP()),
    m_found_entities (),
    m_struct_members (),
    m_keep_result_in_memory (keep_result_in_memory),
    m_parser_vars (),
    m_struct_vars ()
{
    EnableStructVars();
}

ClangExpressionDeclMap::~ClangExpressionDeclMap()
{
    // Note: The model is now that the parser's AST context and all associated
    //   data does not vanish until the expression has been executed.  This means
    //   that valuable lookup data (like namespaces) doesn't vanish, but 
    
    DidParse();
    DisableStructVars();
}

bool 
ClangExpressionDeclMap::WillParse(ExecutionContext &exe_ctx,
                                  Materializer *materializer)
{
    ClangASTMetrics::ClearLocalCounters();
    
    EnableParserVars();
    m_parser_vars->m_exe_ctx = exe_ctx;
    
    Target *target = exe_ctx.GetTargetPtr();
    if (exe_ctx.GetFramePtr())
        m_parser_vars->m_sym_ctx = exe_ctx.GetFramePtr()->GetSymbolContext(lldb::eSymbolContextEverything);
    else if (exe_ctx.GetThreadPtr() && exe_ctx.GetThreadPtr()->GetStackFrameAtIndex(0))
        m_parser_vars->m_sym_ctx = exe_ctx.GetThreadPtr()->GetStackFrameAtIndex(0)->GetSymbolContext(lldb::eSymbolContextEverything);
    else if (exe_ctx.GetProcessPtr())
    {
        m_parser_vars->m_sym_ctx.Clear(true);
        m_parser_vars->m_sym_ctx.target_sp = exe_ctx.GetTargetSP();
    }
    else if (target)
    {
        m_parser_vars->m_sym_ctx.Clear(true);
        m_parser_vars->m_sym_ctx.target_sp = exe_ctx.GetTargetSP();
    }
    
    if (target)
    {
        m_parser_vars->m_persistent_vars = &target->GetPersistentVariables();
    
        if (!target->GetScratchClangASTContext())
            return false;
    }
    
    m_parser_vars->m_target_info = GetTargetInfo();
    m_parser_vars->m_materializer = materializer;
    
    return true;
}

void
ClangExpressionDeclMap::DidParse()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
        ClangASTMetrics::DumpCounters(log);
    
    if (m_parser_vars.get())
    {
        for (size_t entity_index = 0, num_entities = m_found_entities.GetSize();
             entity_index < num_entities;
             ++entity_index)
        {
            ClangExpressionVariableSP var_sp(m_found_entities.GetVariableAtIndex(entity_index));
            if (var_sp)
            {
                ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());
                
                if (parser_vars && parser_vars->m_lldb_value)
                    delete parser_vars->m_lldb_value;
            
                var_sp->DisableParserVars(GetParserID());
            }
        }
        
        for (size_t pvar_index = 0, num_pvars = m_parser_vars->m_persistent_vars->GetSize();
             pvar_index < num_pvars;
             ++pvar_index)
        {
            ClangExpressionVariableSP pvar_sp(m_parser_vars->m_persistent_vars->GetVariableAtIndex(pvar_index));
            if (pvar_sp)
                pvar_sp->DisableParserVars(GetParserID());
        }
        
        DisableParserVars();
    }
}

// Interface for IRForTarget

ClangExpressionDeclMap::TargetInfo 
ClangExpressionDeclMap::GetTargetInfo()
{
    assert (m_parser_vars.get());
    
    TargetInfo ret;
    
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;

    Process *process = exe_ctx.GetProcessPtr();
    if (process)
    {
        ret.byte_order = process->GetByteOrder();
        ret.address_byte_size = process->GetAddressByteSize();
    }
    else 
    {
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
        {
            ret.byte_order = target->GetArchitecture().GetByteOrder();
            ret.address_byte_size = target->GetArchitecture().GetAddressByteSize();
        }
    }

    return ret;
}

bool 
ClangExpressionDeclMap::AddPersistentVariable 
(
    const NamedDecl *decl, 
    const ConstString &name, 
    TypeFromParser parser_type,
    bool is_result,
    bool is_lvalue
)
{
    assert (m_parser_vars.get());
    
    if (m_parser_vars->m_materializer && is_result)
    {
        Error err;
        
        ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
        Target *target = exe_ctx.GetTargetPtr();
        if (target == NULL)
            return false;
        
        ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
        
        TypeFromUser user_type(m_ast_importer->DeportType(context,
                                                          parser_type.GetASTContext(),
                                                          parser_type.GetOpaqueQualType()),
                               context);
        
        uint32_t offset = m_parser_vars->m_materializer->AddResultVariable(user_type, is_lvalue, m_keep_result_in_memory, err);
        
        m_found_entities.CreateVariable(exe_ctx.GetBestExecutionContextScope(),
                                        name,
                                        user_type,
                                        m_parser_vars->m_target_info.byte_order,
                                        m_parser_vars->m_target_info.address_byte_size);
        
        ClangExpressionVariableSP var_sp (m_found_entities.GetVariable(name));
        
        if (!var_sp)
            return false;
        
        var_sp->EnableParserVars(GetParserID());
        
        ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());

        parser_vars->m_named_decl = decl;
        parser_vars->m_parser_type = parser_type;
        
        var_sp->EnableJITVars(GetParserID());
        
        ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID());
        
        jit_vars->m_offset = offset;
        
        return true;
    }
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    Target *target = exe_ctx.GetTargetPtr();
    if (target == NULL)
        return false;

    ASTContext *context(target->GetScratchClangASTContext()->getASTContext());
    
    TypeFromUser user_type(m_ast_importer->DeportType(context, 
                                                      parser_type.GetASTContext(),
                                                      parser_type.GetOpaqueQualType()),
                           context);
    
    if (!user_type.GetOpaqueQualType())
    {
        if (log)
            log->Printf("Persistent variable's type wasn't copied successfully");
        return false;
    }
        
    if (!m_parser_vars->m_target_info.IsValid())
        return false;
    
    if (!m_parser_vars->m_persistent_vars->CreatePersistentVariable (exe_ctx.GetBestExecutionContextScope (),
                                                                     name, 
                                                                     user_type, 
                                                                     m_parser_vars->m_target_info.byte_order,
                                                                     m_parser_vars->m_target_info.address_byte_size))
        return false;
    
    ClangExpressionVariableSP var_sp (m_parser_vars->m_persistent_vars->GetVariable(name));
    
    if (!var_sp)
        return false;
    
    var_sp->m_frozen_sp->SetHasCompleteType();
    
    if (is_result)
        var_sp->m_flags |= ClangExpressionVariable::EVNeedsFreezeDry;
    else
        var_sp->m_flags |= ClangExpressionVariable::EVKeepInTarget; // explicitly-declared persistent variables should persist
    
    if (is_lvalue)
    {
        var_sp->m_flags |= ClangExpressionVariable::EVIsProgramReference;
    }
    else
    {
        var_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
        var_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
    }
    
    if (m_keep_result_in_memory)
    {
        var_sp->m_flags |= ClangExpressionVariable::EVKeepInTarget;
    }
    
    if (log)
        log->Printf("Created persistent variable with flags 0x%hx", var_sp->m_flags);
    
    var_sp->EnableParserVars(GetParserID());
    
    ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());
    
    parser_vars->m_named_decl = decl;
    parser_vars->m_parser_type = parser_type;
    
    return true;
}

bool 
ClangExpressionDeclMap::AddValueToStruct 
(
    const NamedDecl *decl,
    const ConstString &name,
    llvm::Value *value,
    size_t size,
    off_t alignment
)
{
    assert (m_struct_vars.get());
    assert (m_parser_vars.get());
    
    bool is_persistent_variable = false;
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    m_struct_vars->m_struct_laid_out = false;
    
    if (m_struct_members.GetVariable(decl, GetParserID()))
        return true;
    
    ClangExpressionVariableSP var_sp (m_found_entities.GetVariable(decl, GetParserID()));
    
    if (!var_sp)
    {
        var_sp = m_parser_vars->m_persistent_vars->GetVariable(decl, GetParserID());
        is_persistent_variable = true;
    }
    
    if (!var_sp)
        return false;
    
    if (log)
        log->Printf("Adding value for (NamedDecl*)%p [%s - %s] to the structure",
                    decl,
                    name.GetCString(),
                    var_sp->GetName().GetCString());
    
    // We know entity->m_parser_vars is valid because we used a parser variable
    // to find it
    
    ClangExpressionVariable::ParserVars *parser_vars = var_sp->GetParserVars(GetParserID());

    parser_vars->m_llvm_value = value;
    
    if (ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID()))
    {
        // We already laid this out; do not touch
        
        if (log)
            log->Printf("Already placed at 0x%llx", (unsigned long long)jit_vars->m_offset);
    }
    
    var_sp->EnableJITVars(GetParserID());
    
    ClangExpressionVariable::JITVars *jit_vars = var_sp->GetJITVars(GetParserID());

    jit_vars->m_alignment = alignment;
    jit_vars->m_size = size;
    
    m_struct_members.AddVariable(var_sp);
    
    if (m_parser_vars->m_materializer)
    {
        uint32_t offset = 0;

        Error err;

        if (is_persistent_variable)
        {
            offset = m_parser_vars->m_materializer->AddPersistentVariable(var_sp, err);
        }
        else
        {
            if (const lldb_private::Symbol *sym = parser_vars->m_lldb_sym)
                offset = m_parser_vars->m_materializer->AddSymbol(*sym, err);
            else if (const RegisterInfo *reg_info = var_sp->GetRegisterInfo())
                offset = m_parser_vars->m_materializer->AddRegister(*reg_info, err);
            else if (parser_vars->m_lldb_var)
                offset = m_parser_vars->m_materializer->AddVariable(parser_vars->m_lldb_var, err);
        }
        
        if (!err.Success())
            return false;
        
        if (log)
            log->Printf("Placed at 0x%llx", (unsigned long long)offset);
        
        jit_vars->m_offset = offset; // TODO DoStructLayout() should not change this.
    }
    
    return true;
}

bool
ClangExpressionDeclMap::DoStructLayout ()
{
    assert (m_struct_vars.get());
    
    if (m_struct_vars->m_struct_laid_out)
        return true;
    
    if (!m_parser_vars->m_materializer)
        return false;
    
    m_struct_vars->m_struct_alignment = m_parser_vars->m_materializer->GetStructAlignment();
    m_struct_vars->m_struct_size = m_parser_vars->m_materializer->GetStructByteSize();
    m_struct_vars->m_struct_laid_out = true;
    return true;
}

bool ClangExpressionDeclMap::GetStructInfo 
(
    uint32_t &num_elements,
    size_t &size,
    off_t &alignment
)
{
    assert (m_struct_vars.get());
    
    if (!m_struct_vars->m_struct_laid_out)
        return false;
    
    num_elements = m_struct_members.GetSize();
    size = m_struct_vars->m_struct_size;
    alignment = m_struct_vars->m_struct_alignment;
    
    return true;
}

bool 
ClangExpressionDeclMap::GetStructElement 
(
    const NamedDecl *&decl,
    llvm::Value *&value,
    off_t &offset,
    ConstString &name,
    uint32_t index
)
{
    assert (m_struct_vars.get());
    
    if (!m_struct_vars->m_struct_laid_out)
        return false;
    
    if (index >= m_struct_members.GetSize())
        return false;
    
    ClangExpressionVariableSP member_sp(m_struct_members.GetVariableAtIndex(index));
    
    if (!member_sp)
        return false;
    
    ClangExpressionVariable::ParserVars *parser_vars = member_sp->GetParserVars(GetParserID());
    ClangExpressionVariable::JITVars *jit_vars = member_sp->GetJITVars(GetParserID());
    
    if (!parser_vars ||
        !jit_vars ||
        !member_sp->GetValueObject())
        return false;
    
    decl = parser_vars->m_named_decl;
    value = parser_vars->m_llvm_value;
    offset = jit_vars->m_offset;
    name = member_sp->GetName();
        
    return true;
}

bool
ClangExpressionDeclMap::GetFunctionInfo 
(
    const NamedDecl *decl, 
    uint64_t &ptr
)
{
    ClangExpressionVariableSP entity_sp(m_found_entities.GetVariable(decl, GetParserID()));

    if (!entity_sp)
        return false;
    
    // We know m_parser_vars is valid since we searched for the variable by
    // its NamedDecl
    
    ClangExpressionVariable::ParserVars *parser_vars = entity_sp->GetParserVars(GetParserID());

    ptr = parser_vars->m_lldb_value->GetScalar().ULongLong();
    
    return true;
}

static void
FindCodeSymbolInContext
(
    const ConstString &name,
    SymbolContext &sym_ctx,
    SymbolContextList &sc_list
)
{
    SymbolContextList temp_sc_list;
    if (sym_ctx.module_sp)
        sym_ctx.module_sp->FindSymbolsWithNameAndType(name, eSymbolTypeAny, temp_sc_list);
    
    if (!sc_list.GetSize() && sym_ctx.target_sp)
        sym_ctx.target_sp->GetImages().FindSymbolsWithNameAndType(name, eSymbolTypeAny, temp_sc_list);

    unsigned temp_sc_list_size = temp_sc_list.GetSize();
    for (unsigned i = 0; i < temp_sc_list_size; i++)
    {
        SymbolContext sym_ctx;
        temp_sc_list.GetContextAtIndex(i, sym_ctx);
        if (sym_ctx.symbol)
        {
            switch (sym_ctx.symbol->GetType())
            {
                case eSymbolTypeCode:
                case eSymbolTypeResolver:
                    sc_list.Append(sym_ctx);
                    break;

                default:
                    break;
            }
        }
    }
}

bool
ClangExpressionDeclMap::GetFunctionAddress 
(
    const ConstString &name,
    uint64_t &func_addr
)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    ExecutionContext &exe_ctx = m_parser_vars->m_exe_ctx;
    Target *target = exe_ctx.GetTargetPtr();
    // Back out in all cases where we're not fully initialized
    if (target == NULL)
        return false;
    if (!m_parser_vars->m_sym_ctx.target_sp)
        return false;

    SymbolContextList sc_list;
    
    FindCodeSymbolInContext(name, m_parser_vars->m_sym_ctx, sc_list);

    if (!sc_list.GetSize())
    {
        // We occasionally get debug information in which a const function is reported 
        // as non-const, so the mangled name is wrong.  This is a hack to compensate.
        
        if (!strncmp(name.GetCString(), "_ZN", 3) &&
            strncmp(name.GetCString(), "_ZNK", 4))
        {
            std::string fixed_scratch("_ZNK");
            fixed_scratch.append(name.GetCString() + 3);
            ConstString fixed_name(fixed_scratch.c_str());
            
            if (log)
                log->Printf("Failed to find symbols given non-const name %s; trying %s", name.GetCString(), fixed_name.GetCString());
            
            FindCodeSymbolInContext(fixed_name, m_parser_vars->m_sym_ctx, sc_list);
        }
    }
    
    if (!sc_list.GetSize())
        return false;

    SymbolContext sym_ctx;
    sc_list.GetContextAtIndex(0, sym_ctx);

    const Address *func_so_addr = NULL;
    bool is_indirect_function = false;

    if (sym_ctx.function)
        func_so_addr = &sym_ctx.function->GetAddressRange().GetBaseAddress();
    else if (sym_ctx.symbol) {
        func_so_addr = &sym_ctx.symbol->GetAddress();
        is_indirect_function = sym_ctx.symbol->IsIndirect();
    } else
        return false;

    if (!func_so_addr || !func_so_addr->IsValid())
        return false;

    func_addr = func_so_addr->GetCallableLoadAddress (target, is_indirect_function);

    return true;
}

addr_t
ClangExpressionDeclMap::GetSymbolAddress (Target &target, Process *process, const ConstString &name, lldb::SymbolType symbol_type)
{
    SymbolContextList sc_list;
    
    target.GetImages().FindSymbolsWithNameAndType(name, symbol_type, sc_list);
    
    const uint32_t num_matches = sc_list.GetSize();
    addr_t symbol_load_addr = LLDB_INVALID_ADDRESS;

    for (uint32_t i=0; i<num_matches && (symbol_load_addr == 0 || symbol_load_addr == LLDB_INVALID_ADDRESS); i++)
    {
        SymbolContext sym_ctx;
        sc_list.GetContextAtIndex(i, sym_ctx);
    
        const Address *sym_address = &sym_ctx.symbol->GetAddress();
        
        if (!sym_address || !sym_address->IsValid())
            continue;
        
        if (sym_address)
        {
            switch (sym_ctx.symbol->GetType())
            {
                case eSymbolTypeCode:
                case eSymbolTypeTrampoline:
                    symbol_load_addr = sym_address->GetCallableLoadAddress (&target);
                    break;

                case eSymbolTypeResolver:
                    symbol_load_addr = sym_address->GetCallableLoadAddress (&target, true);
                    break;

                case eSymbolTypeData:
                case eSymbolTypeRuntime:
                case eSymbolTypeVariable:
                case eSymbolTypeLocal:
                case eSymbolTypeParam:
                case eSymbolTypeInvalid:
                case eSymbolTypeAbsolute:
                case eSymbolTypeException:
                case eSymbolTypeSourceFile:
                case eSymbolTypeHeaderFile:
                case eSymbolTypeObjectFile:
                case eSymbolTypeCommonBlock:
                case eSymbolTypeBlock:
                case eSymbolTypeVariableType:
                case eSymbolTypeLineEntry:
                case eSymbolTypeLineHeader:
                case eSymbolTypeScopeBegin:
                case eSymbolTypeScopeEnd:
                case eSymbolTypeAdditional:
                case eSymbolTypeCompiler:
                case eSymbolTypeInstrumentation:
                case eSymbolTypeUndefined:
                case eSymbolTypeObjCClass:
                case eSymbolTypeObjCMetaClass:
                case eSymbolTypeObjCIVar:
                    symbol_load_addr = sym_address->GetLoadAddress (&target);
                    break;
            }
        }
    }
    
    if (symbol_load_addr == LLDB_INVALID_ADDRESS && process)
    {
        ObjCLanguageRuntime *runtime = process->GetObjCLanguageRuntime();
        
        if (runtime)
        {
            symbol_load_addr = runtime->LookupRuntimeSymbol(name);
        }
    }
    
    return symbol_load_addr;
}

addr_t
ClangExpressionDeclMap::GetSymbolAddress (const ConstString &name, lldb::SymbolType symbol_type)
{
    assert (m_parser_vars.get());
    
    if (!m_parser_vars->m_exe_ctx.GetTargetPtr())
        return false;
    
    return GetSymbolAddress(m_parser_vars->m_exe_ctx.GetTargetRef(), m_parser_vars->m_exe_ctx.GetProcessPtr(), name, symbol_type);
}

const Symbol *
ClangExpressionDeclMap::FindGlobalDataSymbol (Target &target,
                                              const ConstString &name)
{
    SymbolContextList sc_list;
    
    target.GetImages().FindSymbolsWithNameAndType(name, eSymbolTypeAny, sc_list);
    
    const uint32_t matches = sc_list.GetSize();
    for (uint32_t i=0; i<matches; ++i)
    {
        SymbolContext sym_ctx;
        sc_list.GetContextAtIndex(i, sym_ctx);
        if (sym_ctx.symbol)
        {
            const Symbol *symbol = sym_ctx.symbol;
            const Address *sym_address = &symbol->GetAddress();
            
            if (sym_address && sym_address->IsValid())
            {
                switch (symbol->GetType())
                {
                    case eSymbolTypeData:
                    case eSymbolTypeRuntime:
                    case eSymbolTypeAbsolute:
                    case eSymbolTypeObjCClass:
                    case eSymbolTypeObjCMetaClass:
                    case eSymbolTypeObjCIVar:
                        if (symbol->GetDemangledNameIsSynthesized())
                        {
                            // If the demangled name was synthesized, then don't use it
                            // for expressions. Only let the symbol match if the mangled
                            // named matches for these symbols.
                            if (symbol->GetMangled().GetMangledName() != name)
                                break;
                        }
                        return symbol;

                    case eSymbolTypeCode: // We already lookup functions elsewhere
                    case eSymbolTypeVariable:
                    case eSymbolTypeLocal:
                    case eSymbolTypeParam:
                    case eSymbolTypeTrampoline:
                    case eSymbolTypeInvalid:
                    case eSymbolTypeException:
                    case eSymbolTypeSourceFile:
                    case eSymbolTypeHeaderFile:
                    case eSymbolTypeObjectFile:
                    case eSymbolTypeCommonBlock:
                    case eSymbolTypeBlock:
                    case eSymbolTypeVariableType:
                    case eSymbolTypeLineEntry:
                    case eSymbolTypeLineHeader:
                    case eSymbolTypeScopeBegin:
                    case eSymbolTypeScopeEnd:
                    case eSymbolTypeAdditional:
                    case eSymbolTypeCompiler:
                    case eSymbolTypeInstrumentation:
                    case eSymbolTypeUndefined:
                    case eSymbolTypeResolver:
                        break;
                }
            }
        }
    }
    
    return NULL;
}

lldb::VariableSP
ClangExpressionDeclMap::FindGlobalVariable
(
    Target &target,
    ModuleSP &module,
    const ConstString &name,
    ClangNamespaceDecl *namespace_decl,
    TypeFromUser *type
)
{
    VariableList vars;
    
    if (module && namespace_decl)
        module->FindGlobalVariables (name, namespace_decl, true, -1, vars);
    else
        target.GetImages().FindGlobalVariables(name, true, -1, vars);
    
    if (vars.GetSize())
    {
        if (type)
        {
            for (size_t i = 0; i < vars.GetSize(); ++i)
            {
                VariableSP var_sp = vars.GetVariableAtIndex(i);
                
                if (type->GetASTContext() == var_sp->GetType()->GetClangAST())
                {
                    if (ClangASTContext::AreTypesSame(type->GetASTContext(), type->GetOpaqueQualType(), var_sp->GetType()->GetClangFullType()))
                        return var_sp;
                }
            }
        }
        else
        {
            return vars.GetVariableAtIndex(0);
        }
    }
    
    return VariableSP();
}

// Interface for ClangASTSource

void
ClangExpressionDeclMap::FindExternalVisibleDecls (NameSearchContext &context)
{
    assert (m_ast_context);
    
    ClangASTMetrics::RegisterVisibleQuery();
    
    const ConstString name(context.m_decl_name.getAsString().c_str());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (GetImportInProgress())
    {
        if (log && log->GetVerbose())
            log->Printf("Ignoring a query during an import");
        return;
    }
    
    static unsigned int invocation_id = 0;
    unsigned int current_id = invocation_id++;
    
    if (log)
    {
        if (!context.m_decl_context)
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in a NULL DeclContext", current_id, name.GetCString());
        else if (const NamedDecl *context_named_decl = dyn_cast<NamedDecl>(context.m_decl_context))
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in '%s'", current_id, name.GetCString(), context_named_decl->getNameAsString().c_str());
        else
            log->Printf("ClangExpressionDeclMap::FindExternalVisibleDecls[%u] for '%s' in a '%s'", current_id, name.GetCString(), context.m_decl_context->getDeclKindName());
    }
            
    if (const NamespaceDecl *namespace_context = dyn_cast<NamespaceDecl>(context.m_decl_context))
    {
        ClangASTImporter::NamespaceMapSP namespace_map = m_ast_importer->GetNamespaceMap(namespace_context);
        
        if (log && log->GetVerbose())
            log->Printf("  CEDM::FEVD[%u] Inspecting (NamespaceMap*)%p (%d entries)", 
                        current_id, 
                        namespace_map.get(), 
                        (int)namespace_map->size());
        
        if (!namespace_map)
            return;
        
        for (ClangASTImporter::NamespaceMap::iterator i = namespace_map->begin(), e = namespace_map->end();
             i != e;
             ++i)
        {
            if (log)
                log->Printf("  CEDM::FEVD[%u] Searching namespace %s in module %s",
                            current_id,
                            i->second.GetNamespaceDecl()->getNameAsString().c_str(),
                            i->first->GetFileSpec().GetFilename().GetCString());
                
            FindExternalVisibleDecls(context,
                                     i->first,
                                     i->second,
                                     current_id);
        }
    }
    else if (isa<TranslationUnitDecl>(context.m_decl_context))
    {
        ClangNamespaceDecl namespace_decl;
        
        if (log)
            log->Printf("  CEDM::FEVD[%u] Searching the root namespace", current_id);
        
        FindExternalVisibleDecls(context,
                                 lldb::ModuleSP(),
                                 namespace_decl,
                                 current_id);
    }
    
    if (!context.m_found.variable)
        ClangASTSource::FindExternalVisibleDecls(context);
}

void 
ClangExpressionDeclMap::FindExternalVisibleDecls (NameSearchContext &context, 
                                                  lldb::ModuleSP module_sp,
                                                  ClangNamespaceDecl &namespace_decl,
                                                  unsigned int current_id)
{
    assert (m_ast_context);
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    SymbolContextList sc_list;
    
    const ConstString name(context.m_decl_name.getAsString().c_str());
    
    const char *name_unique_cstr = name.GetCString();
    
    if (name_unique_cstr == NULL)
        return;
    
    static ConstString id_name("id");
    static ConstString Class_name("Class");
    
    if (name == id_name || name == Class_name)
        return;
    
    // Only look for functions by name out in our symbols if the function 
    // doesn't start with our phony prefix of '$'
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();
    StackFrame *frame = m_parser_vars->m_exe_ctx.GetFramePtr();
    if (name_unique_cstr[0] == '$' && !namespace_decl)
    {
        static ConstString g_lldb_class_name ("$__lldb_class");
        
        if (name == g_lldb_class_name)
        {
            // Clang is looking for the type of "this"
                        
            if (frame == NULL)
                return;
            
            SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction);
            
            if (!sym_ctx.function)
                return;
            
            // Get the block that defines the function
            Block *function_block = sym_ctx.GetFunctionBlock();

            if (!function_block)
                return;

            clang::DeclContext *decl_context = function_block->GetClangDeclContext();
            
            if (!decl_context)
                return;
            
            clang::CXXMethodDecl *method_decl = llvm::dyn_cast<clang::CXXMethodDecl>(decl_context);
            
            if (method_decl)
            {
                clang::CXXRecordDecl *class_decl = method_decl->getParent();
                
                QualType class_qual_type(class_decl->getTypeForDecl(), 0);
                
                TypeFromUser class_user_type (class_qual_type.getAsOpaquePtr(),
                                              &class_decl->getASTContext());
                
                if (log)
                {
                    ASTDumper ast_dumper(class_qual_type);
                    log->Printf("  CEDM::FEVD[%u] Adding type for $__lldb_class: %s", current_id, ast_dumper.GetCString());
                }
                
                TypeFromParser class_type = CopyClassType(class_user_type, current_id);
                
                if (!class_type.IsValid())
                    return;
                
                TypeSourceInfo *type_source_info = m_ast_context->getTrivialTypeSourceInfo(QualType::getFromOpaquePtr(class_type.GetOpaqueQualType()));
                
                if (!type_source_info)
                    return;
                
                TypedefDecl *typedef_decl = TypedefDecl::Create(*m_ast_context,
                                                                m_ast_context->getTranslationUnitDecl(),
                                                                SourceLocation(),
                                                                SourceLocation(),
                                                                context.m_decl_name.getAsIdentifierInfo(),
                                                                type_source_info);
                
                
                if (!typedef_decl)
                    return;
                
                context.AddNamedDecl(typedef_decl);
                
                if (method_decl->isInstance())
                {
                    // self is a pointer to the object
                    
                    QualType class_pointer_type = method_decl->getASTContext().getPointerType(class_qual_type);
                    
                    TypeFromUser self_user_type(class_pointer_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                    
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }
            }
            else
            {
                // This branch will get hit if we are executing code in the context of a function that
                // claims to have an object pointer (through DW_AT_object_pointer?) but is not formally a
                // method of the class.  In that case, just look up the "this" variable in the the current
                // scope and use its type.
                // FIXME: This code is formally correct, but clang doesn't currently emit DW_AT_object_pointer
                // for C++ so it hasn't actually been tested.
                
                VariableList *vars = frame->GetVariableList(false);
                
                lldb::VariableSP this_var = vars->FindVariable(ConstString("this"));
                
                if (this_var &&
                    this_var->IsInScope(frame) &&
                    this_var->LocationIsValidForFrame (frame))
                {
                    Type *this_type = this_var->GetType();
                    
                    if (!this_type)
                        return;
                    
                    QualType this_qual_type = QualType::getFromOpaquePtr(this_type->GetClangFullType());
                    const PointerType *class_pointer_type = this_qual_type->getAs<PointerType>();
                    
                    if (class_pointer_type)
                    {
                        QualType class_type = class_pointer_type->getPointeeType();
                        
                        if (!class_type.getAsOpaquePtr())
                            return;
                        
                        if (log)
                        {
                            ASTDumper ast_dumper(this_type->GetClangFullType());
                            log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                        }
                        
                        TypeFromUser class_user_type (class_type.getAsOpaquePtr(),
                                                        this_type->GetClangAST());
                        AddOneType(context, class_user_type, current_id);
                                    
                                    
                        TypeFromUser this_user_type(this_type->GetClangFullType(),
                                                    this_type->GetClangAST());
                        
                        m_struct_vars->m_object_pointer_type = this_user_type;
                        return;
                    }
                }
            }
            
            return;
        }
        
        static ConstString g_lldb_objc_class_name ("$__lldb_objc_class");
        if (name == g_lldb_objc_class_name)
        {
            // Clang is looking for the type of "*self"
            
            if (!frame)
                return;
         
            SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction);
            
            if (!sym_ctx.function)
                return;
            
            // Get the block that defines the function
            Block *function_block = sym_ctx.GetFunctionBlock();
            
            if (!function_block)
                return;
            
            clang::DeclContext *decl_context = function_block->GetClangDeclContext();
            
            if (!decl_context)
                return;
            
            clang::ObjCMethodDecl *method_decl = llvm::dyn_cast<clang::ObjCMethodDecl>(decl_context);
            
            if (method_decl)
            {
                ObjCInterfaceDecl* self_interface = method_decl->getClassInterface();
                
                if (!self_interface)
                    return;
                
                const clang::Type *interface_type = self_interface->getTypeForDecl();
                
                if (!interface_type)
                    return; // This is unlikely, but we have seen crashes where this occurred
                        
                TypeFromUser class_user_type(QualType(interface_type, 0).getAsOpaquePtr(),
                                             &method_decl->getASTContext());
                
                if (log)
                {
                    ASTDumper ast_dumper(interface_type);
                    log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                }
                    
                AddOneType(context, class_user_type, current_id);
                                
                if (method_decl->isInstanceMethod())
                {
                    // self is a pointer to the object
                    
                    QualType class_pointer_type = method_decl->getASTContext().getObjCObjectPointerType(QualType(interface_type, 0));
                
                    TypeFromUser self_user_type(class_pointer_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }
                else
                {
                    // self is a Class pointer
                    QualType class_type = method_decl->getASTContext().getObjCClassType();
                    
                    TypeFromUser self_user_type(class_type.getAsOpaquePtr(),
                                                &method_decl->getASTContext());
                    
                    m_struct_vars->m_object_pointer_type = self_user_type;
                }

                return;
            }
            else
            {
                // This branch will get hit if we are executing code in the context of a function that
                // claims to have an object pointer (through DW_AT_object_pointer?) but is not formally a
                // method of the class.  In that case, just look up the "self" variable in the the current
                // scope and use its type.
                
                VariableList *vars = frame->GetVariableList(false);
                
                lldb::VariableSP self_var = vars->FindVariable(ConstString("self"));
                
                if (self_var &&
                    self_var->IsInScope(frame) && 
                    self_var->LocationIsValidForFrame (frame))
                {
                    Type *self_type = self_var->GetType();
                    
                    if (!self_type)
                        return;
                    
                    QualType self_qual_type = QualType::getFromOpaquePtr(self_type->GetClangFullType());
                    
                    if (self_qual_type->isObjCClassType())
                    {
                        return;
                    }
                    else if (self_qual_type->isObjCObjectPointerType())
                    {
                        const ObjCObjectPointerType *class_pointer_type = self_qual_type->getAs<ObjCObjectPointerType>();
                    
                        QualType class_type = class_pointer_type->getPointeeType();
                        
                        if (!class_type.getAsOpaquePtr())
                            return;
                        
                        if (log)
                        {
                            ASTDumper ast_dumper(self_type->GetClangFullType());
                            log->Printf("  FEVD[%u] Adding type for $__lldb_objc_class: %s", current_id, ast_dumper.GetCString());
                        }
                        
                        TypeFromUser class_user_type (class_type.getAsOpaquePtr(),
                                                      self_type->GetClangAST());
                        
                        AddOneType(context, class_user_type, current_id);
                                    
                        TypeFromUser self_user_type(self_type->GetClangFullType(),
                                                    self_type->GetClangAST());
                        
                        m_struct_vars->m_object_pointer_type = self_user_type;
                        return;
                    }
                }
            }

            return;
        }
        
        // any other $__lldb names should be weeded out now
        if (!::strncmp(name_unique_cstr, "$__lldb", sizeof("$__lldb") - 1))
            return;
        
        do
        {
            if (!target)
                break;
            
            ClangASTContext *scratch_clang_ast_context = target->GetScratchClangASTContext();
            
            if (!scratch_clang_ast_context)
                break;
            
            ASTContext *scratch_ast_context = scratch_clang_ast_context->getASTContext();
            
            if (!scratch_ast_context)
                break;
            
            TypeDecl *ptype_type_decl = m_parser_vars->m_persistent_vars->GetPersistentType(name);
            
            if (!ptype_type_decl)
                break;
            
            Decl *parser_ptype_decl = m_ast_importer->CopyDecl(m_ast_context, scratch_ast_context, ptype_type_decl);
            
            if (!parser_ptype_decl)
                break;
            
            TypeDecl *parser_ptype_type_decl = dyn_cast<TypeDecl>(parser_ptype_decl);
            
            if (!parser_ptype_type_decl)
                break;
            
            if (log)
                log->Printf("  CEDM::FEVD[%u] Found persistent type %s", current_id, name.GetCString());
            
            context.AddNamedDecl(parser_ptype_type_decl);
        } while (0);
        
        ClangExpressionVariableSP pvar_sp(m_parser_vars->m_persistent_vars->GetVariable(name));
        
        if (pvar_sp)
        {
            AddOneVariable(context, pvar_sp, current_id);
            return;
        }
        
        const char *reg_name(&name.GetCString()[1]);
        
        if (m_parser_vars->m_exe_ctx.GetRegisterContext())
        {
            const RegisterInfo *reg_info(m_parser_vars->m_exe_ctx.GetRegisterContext()->GetRegisterInfoByName(reg_name));
            
            if (reg_info)
            {
                if (log)
                    log->Printf("  CEDM::FEVD[%u] Found register %s", current_id, reg_info->name);
                
                AddOneRegister(context, reg_info, current_id);
            }
        }
    }
    else
    {
        ValueObjectSP valobj;
        VariableSP var;
        Error err;
        
        if (frame && !namespace_decl)
        {
            valobj = frame->GetValueForVariableExpressionPath(name_unique_cstr, 
                                                              eNoDynamicValues, 
                                                              StackFrame::eExpressionPathOptionCheckPtrVsMember ||
                                                              StackFrame::eExpressionPathOptionsAllowDirectIVarAccess ||
                                                              StackFrame::eExpressionPathOptionsNoFragileObjcIvar ||
                                                              StackFrame::eExpressionPathOptionsNoSyntheticChildren ||
                                                              StackFrame::eExpressionPathOptionsNoSyntheticArrayRange,
                                                              var,
                                                              err);
            
            // If we found a variable in scope, no need to pull up function names
            if (err.Success() && var)
            {
                AddOneVariable(context, var, valobj, current_id);
                context.m_found.variable = true;
                return;
            }
        }
        
        if (target)
        {
            var = FindGlobalVariable (*target,
                                      module_sp,
                                      name,
                                      &namespace_decl,
                                      NULL);
            
            if (var)
            {
                valobj = ValueObjectVariable::Create(target, var);
                AddOneVariable(context, var, valobj, current_id);
                context.m_found.variable = true;
                return;
            }
        }
        
        if (!context.m_found.variable)
        {
            const bool include_inlines = false;
            const bool append = false;
            
            if (namespace_decl && module_sp)
            {
                const bool include_symbols = false;

                module_sp->FindFunctions(name,
                                         &namespace_decl,
                                         eFunctionNameTypeBase, 
                                         include_symbols,
                                         include_inlines,
                                         append,
                                         sc_list);
            }
            else if (target && !namespace_decl)
            {
                const bool include_symbols = true;
                
                // TODO Fix FindFunctions so that it doesn't return
                //   instance methods for eFunctionNameTypeBase.
                
                target->GetImages().FindFunctions(name,
                                                  eFunctionNameTypeFull,
                                                  include_symbols,
                                                  include_inlines,
                                                  append, 
                                                  sc_list);
            }
            
            if (sc_list.GetSize())
            {
                Symbol *generic_symbol = NULL;
                Symbol *non_extern_symbol = NULL;
                
                for (uint32_t index = 0, num_indices = sc_list.GetSize();
                     index < num_indices;
                     ++index)
                {
                    SymbolContext sym_ctx;
                    sc_list.GetContextAtIndex(index, sym_ctx);
                    
                    if (sym_ctx.function)
                    {
                        clang::DeclContext *decl_ctx = sym_ctx.function->GetClangDeclContext();
                        
                        if (!decl_ctx)
                            continue;
                        
                        // Filter out class/instance methods.
                        if (dyn_cast<clang::ObjCMethodDecl>(decl_ctx))
                            continue;
                        if (dyn_cast<clang::CXXMethodDecl>(decl_ctx))
                            continue;
                        
                        AddOneFunction(context, sym_ctx.function, NULL, current_id);
                        context.m_found.function_with_type_info = true;
                        context.m_found.function = true;
                    }
                    else if (sym_ctx.symbol)
                    {
                        if (sym_ctx.symbol->IsExternal())
                            generic_symbol = sym_ctx.symbol;
                        else
                            non_extern_symbol = sym_ctx.symbol;
                    }
                }
                
                if (!context.m_found.function_with_type_info)
                {
                    if (generic_symbol)
                    {
                        AddOneFunction (context, NULL, generic_symbol, current_id);
                        context.m_found.function = true;
                    }
                    else if (non_extern_symbol)
                    {
                        AddOneFunction (context, NULL, non_extern_symbol, current_id);
                        context.m_found.function = true;
                    }
                }
            }
            
            if (target && !context.m_found.variable && !namespace_decl)
            {
                // We couldn't find a non-symbol variable for this.  Now we'll hunt for a generic 
                // data symbol, and -- if it is found -- treat it as a variable.
                
                const Symbol *data_symbol = FindGlobalDataSymbol(*target, name);
                
                if (data_symbol)
                {
                    AddOneGenericVariable(context, *data_symbol, current_id);
                    context.m_found.variable = true;
                }
            }
        }
    }
}

static clang_type_t
MaybePromoteToBlockPointerType
(
    ASTContext *ast_context,
    clang_type_t candidate_type
)
{
    if (!candidate_type)
        return candidate_type;
    
    QualType candidate_qual_type = QualType::getFromOpaquePtr(candidate_type);
    
    const PointerType *candidate_pointer_type = dyn_cast<PointerType>(candidate_qual_type);
    
    if (!candidate_pointer_type)
        return candidate_type;
    
    QualType pointee_qual_type = candidate_pointer_type->getPointeeType();
    
    const RecordType *pointee_record_type = dyn_cast<RecordType>(pointee_qual_type);
    
    if (!pointee_record_type)
        return candidate_type;
    
    RecordDecl *pointee_record_decl = pointee_record_type->getDecl();
    
    if (!pointee_record_decl->isRecord())
        return candidate_type;
    
    if (!pointee_record_decl->getName().startswith(llvm::StringRef("__block_literal_")))
        return candidate_type;
    
    QualType generic_function_type = ast_context->getFunctionNoProtoType(ast_context->UnknownAnyTy);
    QualType block_pointer_type = ast_context->getBlockPointerType(generic_function_type);
    
    return block_pointer_type.getAsOpaquePtr();
}

Value *
ClangExpressionDeclMap::GetVariableValue
(
    VariableSP &var,
    ASTContext *parser_ast_context,
    TypeFromUser *user_type,
    TypeFromParser *parser_type
)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Type *var_type = var->GetType();
    
    if (!var_type) 
    {
        if (log)
            log->PutCString("Skipped a definition because it has no type");
        return NULL;
    }
    
    clang_type_t var_opaque_type = var_type->GetClangFullType();
    
    if (!var_opaque_type)
    {
        if (log)
            log->PutCString("Skipped a definition because it has no Clang type");
        return NULL;
    }
    
    ASTContext *ast = var_type->GetClangASTContext().getASTContext();
    
    if (!ast)
    {
        if (log)
            log->PutCString("There is no AST context for the current execution context");
        return NULL;
    }

    // commented out because of <rdar://problem/11024417>
    //var_opaque_type = MaybePromoteToBlockPointerType (ast, var_opaque_type);
    
    DWARFExpression &var_location_expr = var->LocationExpression();
    
    std::unique_ptr<Value> var_location(new Value);
    
    lldb::addr_t loclist_base_load_addr = LLDB_INVALID_ADDRESS;
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    if (var_location_expr.IsLocationList())
    {
        SymbolContext var_sc;
        var->CalculateSymbolContext (&var_sc);
        loclist_base_load_addr = var_sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress (target);
    }
    Error err;
    
    if (var->GetLocationIsConstantValueData())
    {
        DataExtractor const_value_extractor;
        
        if (var_location_expr.GetExpressionData(const_value_extractor))
        {
            var_location->operator=(Value(const_value_extractor.GetDataStart(), const_value_extractor.GetByteSize()));
            var_location->SetValueType(Value::eValueTypeHostAddress);
        }
        else
        {
            if (log)
                log->Printf("Error evaluating constant variable: %s", err.AsCString());
            return NULL;
        }
    }
    
    void *type_to_use = NULL;
    
    if (parser_ast_context)
    {
        type_to_use = GuardedCopyType(parser_ast_context, ast, var_opaque_type);
        
        if (!type_to_use)
        {
            if (log)
                log->Printf("Couldn't copy a variable's type into the parser's AST context");
            
            return NULL;
        }
        
        if (parser_type)
            *parser_type = TypeFromParser(type_to_use, parser_ast_context);
    }
    else
        type_to_use = var_opaque_type;
    
    if (var_location.get()->GetContextType() == Value::eContextTypeInvalid)
        var_location.get()->SetContext(Value::eContextTypeClangType, type_to_use);
    
    if (var_location.get()->GetValueType() == Value::eValueTypeFileAddress)
    {
        SymbolContext var_sc;
        var->CalculateSymbolContext(&var_sc);
        
        if (!var_sc.module_sp)
            return NULL;
        
        ObjectFile *object_file = var_sc.module_sp->GetObjectFile();
        
        if (!object_file)
            return NULL;
        
        Address so_addr(var_location->GetScalar().ULongLong(), object_file->GetSectionList());
        
        lldb::addr_t load_addr = so_addr.GetLoadAddress(target);
        
        if (load_addr != LLDB_INVALID_ADDRESS)
        {
            var_location->GetScalar() = load_addr;
            var_location->SetValueType(Value::eValueTypeLoadAddress);
        }
    }
    
    if (user_type)
        *user_type = TypeFromUser(var_opaque_type, ast);
    
    return var_location.release();
}

void
ClangExpressionDeclMap::AddOneVariable (NameSearchContext &context, VariableSP var, ValueObjectSP valobj, unsigned int current_id)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
    TypeFromUser ut;
    TypeFromParser pt;
    
    Value *var_location = GetVariableValue (var, 
                                            m_ast_context,
                                            &ut,
                                            &pt);
    
    clang::QualType parser_opaque_type = QualType::getFromOpaquePtr(pt.GetOpaqueQualType());
    
    if (parser_opaque_type.isNull())
        return;
    
    if (const clang::Type *parser_type = parser_opaque_type.getTypePtr())
    {
        if (const TagType *tag_type = dyn_cast<TagType>(parser_type))
            CompleteType(tag_type->getDecl());
    }
    
    if (!var_location)
        return;
    
    NamedDecl *var_decl;
    
    bool is_reference = ClangASTContext::IsReferenceType(pt.GetOpaqueQualType());

    if (is_reference)
        var_decl = context.AddVarDecl(pt.GetOpaqueQualType());
    else
        var_decl = context.AddVarDecl(ClangASTContext::CreateLValueReferenceType(pt.GetASTContext(), pt.GetOpaqueQualType()));
        
    std::string decl_name(context.m_decl_name.getAsString());
    ConstString entity_name(decl_name.c_str());
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (valobj));
    
    assert (entity.get());
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = pt;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = var_location;
    parser_vars->m_lldb_var    = var;
    
    if (is_reference)
        entity->m_flags |= ClangExpressionVariable::EVTypeIsReference;
    
    if (log)
    {
        ASTDumper orig_dumper(ut.GetOpaqueQualType());
        ASTDumper ast_dumper(var_decl);        
        log->Printf("  CEDM::FEVD[%u] Found variable %s, returned %s (original %s)", current_id, decl_name.c_str(), ast_dumper.GetCString(), orig_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneVariable(NameSearchContext &context,
                                       ClangExpressionVariableSP &pvar_sp, 
                                       unsigned int current_id)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    TypeFromUser user_type (pvar_sp->GetTypeFromUser());
    
    TypeFromParser parser_type (GuardedCopyType(m_ast_context, 
                                                user_type.GetASTContext(), 
                                                user_type.GetOpaqueQualType()),
                                m_ast_context);
    
    if (!parser_type.GetOpaqueQualType())
    {
        if (log)
            log->Printf("  CEDM::FEVD[%u] Couldn't import type for pvar %s", current_id, pvar_sp->GetName().GetCString());
        return;
    }
    
    NamedDecl *var_decl = context.AddVarDecl(ClangASTContext::CreateLValueReferenceType(parser_type.GetASTContext(), parser_type.GetOpaqueQualType()));
    
    pvar_sp->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = pvar_sp->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = NULL;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        log->Printf("  CEDM::FEVD[%u] Added pvar %s, returned %s", current_id, pvar_sp->GetName().GetCString(), ast_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneGenericVariable(NameSearchContext &context, 
                                              const Symbol &symbol,
                                              unsigned int current_id)
{
    assert(m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    if (target == NULL)
        return;

    ASTContext *scratch_ast_context = target->GetScratchClangASTContext()->getASTContext();
    
    TypeFromUser user_type (ClangASTContext::CreateLValueReferenceType(scratch_ast_context, ClangASTContext::GetVoidPtrType(scratch_ast_context, false)),
                            scratch_ast_context);
    
    TypeFromParser parser_type (ClangASTContext::CreateLValueReferenceType(m_ast_context, ClangASTContext::GetVoidPtrType(m_ast_context, false)),
                                m_ast_context);
    
    NamedDecl *var_decl = context.AddVarDecl(parser_type.GetOpaqueQualType());
    
    std::string decl_name(context.m_decl_name.getAsString());
    ConstString entity_name(decl_name.c_str());
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope (),
                                                                      entity_name, 
                                                                      user_type,
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    
    std::unique_ptr<Value> symbol_location(new Value);
    
    const Address &symbol_address = symbol.GetAddress();
    lldb::addr_t symbol_load_addr = symbol_address.GetLoadAddress(target);
    
    symbol_location->SetContext(Value::eContextTypeClangType, user_type.GetOpaqueQualType());
    symbol_location->GetScalar() = symbol_load_addr;
    symbol_location->SetValueType(Value::eValueTypeLoadAddress);
    
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = symbol_location.release();
    parser_vars->m_lldb_sym    = &symbol;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        
        log->Printf("  CEDM::FEVD[%u] Found variable %s, returned %s", current_id, decl_name.c_str(), ast_dumper.GetCString());
    }
}

bool 
ClangExpressionDeclMap::ResolveUnknownTypes()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    ASTContext *scratch_ast_context = target->GetScratchClangASTContext()->getASTContext();

    for (size_t index = 0, num_entities = m_found_entities.GetSize();
         index < num_entities;
         ++index)
    {
        ClangExpressionVariableSP entity = m_found_entities.GetVariableAtIndex(index);
        
        ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
        
        if (entity->m_flags & ClangExpressionVariable::EVUnknownType)
        {
            const NamedDecl *named_decl = parser_vars->m_named_decl;
            const VarDecl *var_decl = dyn_cast<VarDecl>(named_decl);
            
            if (!var_decl)
            {
                if (log)
                    log->Printf("Entity of unknown type does not have a VarDecl");
                return false;
            }
            
            if (log)
            {
                ASTDumper ast_dumper(const_cast<VarDecl*>(var_decl));
                log->Printf("Variable of unknown type now has Decl %s", ast_dumper.GetCString());
            }
                
            QualType var_type = var_decl->getType();
            TypeFromParser parser_type(var_type.getAsOpaquePtr(), &var_decl->getASTContext());
            
            lldb::clang_type_t copied_type = m_ast_importer->CopyType(scratch_ast_context, &var_decl->getASTContext(), var_type.getAsOpaquePtr());
            
            if (!copied_type)
            {                
                if (log)
                    log->Printf("ClangExpressionDeclMap::ResolveUnknownType - Couldn't import the type for a variable");
                
                return (bool) lldb::ClangExpressionVariableSP();
            }
            
            TypeFromUser user_type(copied_type, scratch_ast_context);
                        
            parser_vars->m_lldb_value->SetContext(Value::eContextTypeClangType, user_type.GetOpaqueQualType());
            parser_vars->m_parser_type = parser_type;
            
            entity->SetClangAST(user_type.GetASTContext());
            entity->SetClangType(user_type.GetOpaqueQualType());
            
            entity->m_flags &= ~(ClangExpressionVariable::EVUnknownType);
        }
    }
            
    return true;
}

void
ClangExpressionDeclMap::AddOneRegister (NameSearchContext &context,
                                        const RegisterInfo *reg_info, 
                                        unsigned int current_id)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    void *ast_type = ClangASTContext::GetBuiltinTypeForEncodingAndBitSize(m_ast_context,
                                                                          reg_info->encoding,
                                                                          reg_info->byte_size * 8);
    
    if (!ast_type)
    {
        if (log)
            log->Printf("  Tried to add a type for %s, but couldn't get one", context.m_decl_name.getAsString().c_str());
        return;
    }
    
    TypeFromParser parser_type (ast_type,
                                m_ast_context);
    
    NamedDecl *var_decl = context.AddVarDecl(parser_type.GetOpaqueQualType());
    
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope(),
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    
    std::string decl_name(context.m_decl_name.getAsString());
    entity->SetName (ConstString (decl_name.c_str()));
    entity->SetRegisterInfo (reg_info);
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_parser_type = parser_type;
    parser_vars->m_named_decl  = var_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = NULL;
    entity->m_flags |= ClangExpressionVariable::EVBareRegister;
    
    if (log)
    {
        ASTDumper ast_dumper(var_decl);
        log->Printf("  CEDM::FEVD[%d] Added register %s, returned %s", current_id, context.m_decl_name.getAsString().c_str(), ast_dumper.GetCString());
    }
}

void
ClangExpressionDeclMap::AddOneFunction (NameSearchContext &context,
                                        Function* fun,
                                        Symbol* symbol,
                                        unsigned int current_id)
{
    assert (m_parser_vars.get());
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    NamedDecl *fun_decl = NULL;
    std::unique_ptr<Value> fun_location(new Value);
    const Address *fun_address = NULL;
    
    // only valid for Functions, not for Symbols
    void *fun_opaque_type = NULL;
    ASTContext *fun_ast_context = NULL;

    bool is_indirect_function = false;

    if (fun)
    {
        Type *fun_type = fun->GetType();
        
        if (!fun_type) 
        {
            if (log)
                log->PutCString("  Skipped a function because it has no type");
            return;
        }
        
        fun_opaque_type = fun_type->GetClangFullType();
        
        if (!fun_opaque_type)
        {
            if (log)
                log->PutCString("  Skipped a function because it has no Clang type");
            return;
        }
        
        fun_address = &fun->GetAddressRange().GetBaseAddress();
        
        fun_ast_context = fun_type->GetClangASTContext().getASTContext();
        void *copied_type = GuardedCopyType(m_ast_context, fun_ast_context, fun_opaque_type);
        if (copied_type)
        {
            fun_decl = context.AddFunDecl(copied_type);
            
            if (!fun_decl)
            {
                if (log)
                {
                    log->Printf ("  Failed to create a function decl for '%s' {0x%8.8" PRIx64 "}",
                                 fun_type->GetName().GetCString(),
                                 fun_type->GetID());
                }
                
                return;
            }
        }
        else
        {
            // We failed to copy the type we found
            if (log)
            {
                log->Printf ("  Failed to import the function type '%s' {0x%8.8" PRIx64 "} into the expression parser AST contenxt",
                             fun_type->GetName().GetCString(), 
                             fun_type->GetID());
            }
            
            return;
        }
    }
    else if (symbol)
    {
        fun_address = &symbol->GetAddress();
        fun_decl = context.AddGenericFunDecl();
        is_indirect_function = symbol->IsIndirect();
    }
    else
    {
        if (log)
            log->PutCString("  AddOneFunction called with no function and no symbol");
        return;
    }
    
    Target *target = m_parser_vars->m_exe_ctx.GetTargetPtr();

    lldb::addr_t load_addr = fun_address->GetCallableLoadAddress(target, is_indirect_function);
    
    if (load_addr != LLDB_INVALID_ADDRESS)
    {
        fun_location->SetValueType(Value::eValueTypeLoadAddress);
        fun_location->GetScalar() = load_addr;
    }
    else
    {
        // We have to try finding a file address.
        
        lldb::addr_t file_addr = fun_address->GetFileAddress();
        
        fun_location->SetValueType(Value::eValueTypeFileAddress);
        fun_location->GetScalar() = file_addr;
    }
    
    ClangExpressionVariableSP entity(m_found_entities.CreateVariable (m_parser_vars->m_exe_ctx.GetBestExecutionContextScope (),
                                                                      m_parser_vars->m_target_info.byte_order,
                                                                      m_parser_vars->m_target_info.address_byte_size));
    assert (entity.get());
    std::string decl_name(context.m_decl_name.getAsString());
    entity->SetName(ConstString(decl_name.c_str()));
    entity->SetClangType (fun_opaque_type);
    entity->SetClangAST (fun_ast_context);
    
    entity->EnableParserVars(GetParserID());
    ClangExpressionVariable::ParserVars *parser_vars = entity->GetParserVars(GetParserID());
    parser_vars->m_named_decl  = fun_decl;
    parser_vars->m_llvm_value  = NULL;
    parser_vars->m_lldb_value  = fun_location.release();
        
    if (log)
    {
        ASTDumper ast_dumper(fun_decl);
        
        StreamString ss;
        
        fun_address->Dump(&ss, m_parser_vars->m_exe_ctx.GetBestExecutionContextScope(), Address::DumpStyleResolvedDescription);
        
        log->Printf("  CEDM::FEVD[%u] Found %s function %s (description %s), returned %s",
                    current_id,
                    (fun ? "specific" : "generic"), 
                    decl_name.c_str(),
                    ss.GetData(),
                    ast_dumper.GetCString());
    }
}

TypeFromParser
ClangExpressionDeclMap::CopyClassType(TypeFromUser &ut,
                                      unsigned int current_id)
{
    ASTContext *parser_ast_context = m_ast_context;
    ASTContext *user_ast_context = ut.GetASTContext();

    void *copied_type = GuardedCopyType(parser_ast_context, user_ast_context, ut.GetOpaqueQualType());
    
    if (!copied_type)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
            log->Printf("ClangExpressionDeclMap::CopyClassType - Couldn't import the type");
        
        return TypeFromParser();
    }

    if (ClangASTContext::IsAggregateType(copied_type) && ClangASTContext::GetCompleteType (parser_ast_context, copied_type))
    {
        void *args[1];
        
        args[0] = ClangASTContext::GetVoidPtrType(parser_ast_context, false);
        
        clang_type_t method_type = ClangASTContext::CreateFunctionType (parser_ast_context,
                                                                        ClangASTContext::GetBuiltInType_void(parser_ast_context),
                                                                        args,
                                                                        1,
                                                                        false,
                                                                        ClangASTContext::GetTypeQualifiers(copied_type));
        
        const bool is_virtual = false;
        const bool is_static = false;
        const bool is_inline = false;
        const bool is_explicit = false;
        const bool is_attr_used = true;
        const bool is_artificial = false;
        
        ClangASTContext::AddMethodToCXXRecordType (parser_ast_context,
                                                   copied_type,
                                                   "$__lldb_expr",
                                                   method_type,
                                                   lldb::eAccessPublic,
                                                   is_virtual,
                                                   is_static,
                                                   is_inline,
                                                   is_explicit,
                                                   is_attr_used,
                                                   is_artificial);
    }
    
    return TypeFromParser(copied_type, parser_ast_context);
}

void 
ClangExpressionDeclMap::AddOneType(NameSearchContext &context, 
                                   TypeFromUser &ut,
                                   unsigned int current_id)
{
    ASTContext *parser_ast_context = m_ast_context;
    ASTContext *user_ast_context = ut.GetASTContext();
    
    void *copied_type = GuardedCopyType(parser_ast_context, user_ast_context, ut.GetOpaqueQualType());
    
    if (!copied_type)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
            log->Printf("ClangExpressionDeclMap::AddOneType - Couldn't import the type");
        
        return;
    }
    
    context.AddTypeDecl(copied_type);
}
