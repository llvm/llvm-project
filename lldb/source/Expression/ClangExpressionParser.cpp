//===-- ClangExpressionParser.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionParser.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/IRDynamicChecks.h"
#include "lldb/Expression/RecordingMemoryManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"

#if defined(__FreeBSD__)
#define USE_STANDARD_JIT
#endif

#if defined (USE_STANDARD_JIT)
#include "llvm/ExecutionEngine/JIT.h"
#else
#include "llvm/ExecutionEngine/MCJIT.h"
#endif
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"

using namespace clang;
using namespace llvm;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// Utility Methods for Clang
//===----------------------------------------------------------------------===//

std::string GetBuiltinIncludePath(const char *Argv0) {
    llvm::sys::Path P =
    llvm::sys::Path::GetMainExecutable(Argv0,
                                       (void*)(intptr_t) GetBuiltinIncludePath);
    
    if (!P.isEmpty()) {
        P.eraseComponent();  // Remove /clang from foo/bin/clang
        P.eraseComponent();  // Remove /bin   from foo/bin
        
        // Get foo/lib/clang/<version>/include
        P.appendComponent("lib");
        P.appendComponent("clang");
        P.appendComponent(CLANG_VERSION_STRING);
        P.appendComponent("include");
    }
    
    return P.str();
}


//===----------------------------------------------------------------------===//
// Main driver for Clang
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
    DiagnosticsEngine &Diags = *static_cast<DiagnosticsEngine*>(UserData);
    
    Diags.Report(diag::err_fe_error_backend) << Message;
    
    // We cannot recover from llvm errors.
    assert(0);
}

static FrontendAction *CreateFrontendBaseAction(CompilerInstance &CI) {
    using namespace clang::frontend;
    
    switch (CI.getFrontendOpts().ProgramAction) {
        default:
            llvm_unreachable("Invalid program action!");
            
        case ASTDump:                return new ASTDumpAction();
        case ASTPrint:               return new ASTPrintAction();
        case ASTDumpXML:             return new ASTDumpXMLAction();
        case ASTView:                return new ASTViewAction();
        case DumpRawTokens:          return new DumpRawTokensAction();
        case DumpTokens:             return new DumpTokensAction();
        case EmitAssembly:           return new EmitAssemblyAction();
        case EmitBC:                 return new EmitBCAction();
        case EmitHTML:               return new HTMLPrintAction();
        case EmitLLVM:               return new EmitLLVMAction();
        case EmitLLVMOnly:           return new EmitLLVMOnlyAction();
        case EmitCodeGenOnly:        return new EmitCodeGenOnlyAction();
        case EmitObj:                return new EmitObjAction();
        case FixIt:                  return new FixItAction();
        case GeneratePCH:            return new GeneratePCHAction();
        case GeneratePTH:            return new GeneratePTHAction();
        case InitOnly:               return new InitOnlyAction();
        case ParseSyntaxOnly:        return new SyntaxOnlyAction();
            
        case PluginAction: {
            for (FrontendPluginRegistry::iterator it =
                 FrontendPluginRegistry::begin(), ie = FrontendPluginRegistry::end();
                 it != ie; ++it) {
                if (it->getName() == CI.getFrontendOpts().ActionName) {
                    llvm::OwningPtr<PluginASTAction> P(it->instantiate());
                    if (!P->ParseArgs(CI, CI.getFrontendOpts().PluginArgs))
                        return 0;
                    return P.take();
                }
            }
            
            CI.getDiagnostics().Report(diag::err_fe_invalid_plugin_name)
            << CI.getFrontendOpts().ActionName;
            return 0;
        }
            
        case PrintDeclContext:       return new DeclContextPrintAction();
        case PrintPreamble:          return new PrintPreambleAction();
        case PrintPreprocessedInput: return new PrintPreprocessedAction();
        case RewriteMacros:          return new RewriteMacrosAction();
        case RewriteObjC:            return new RewriteObjCAction();
        case RewriteTest:            return new RewriteTestAction();
        //case RunAnalysis:            return new AnalysisAction();
        case RunPreprocessorOnly:    return new PreprocessOnlyAction();
    }
}

static FrontendAction *CreateFrontendAction(CompilerInstance &CI) {
    // Create the underlying action.
    FrontendAction *Act = CreateFrontendBaseAction(CI);
    if (!Act)
        return 0;
    
    // If there are any AST files to merge, create a frontend action
    // adaptor to perform the merge.
    if (!CI.getFrontendOpts().ASTMergeFiles.empty())
        Act = new ASTMergeAction(Act, CI.getFrontendOpts().ASTMergeFiles);
    
    return Act;
}

//===----------------------------------------------------------------------===//
// Implementation of ClangExpressionParser
//===----------------------------------------------------------------------===//

ClangExpressionParser::ClangExpressionParser (ExecutionContextScope *exe_scope,
                                              ClangExpression &expr) :
    m_expr (expr),
    m_compiler (),
    m_code_generator (NULL),
    m_jitted_functions ()
{
    // Initialize targets first, so that --version shows registered targets.
    static struct InitializeLLVM {
        InitializeLLVM() {
            llvm::InitializeAllTargets();
            llvm::InitializeAllAsmPrinters();
            llvm::InitializeAllTargetMCs();
            llvm::InitializeAllDisassemblers();
            
            llvm::DisablePrettyStackTrace = true;
        }
    } InitializeLLVM;
    
    // 1. Create a new compiler instance.
    m_compiler.reset(new CompilerInstance());    
    
    // 2. Install the target.

    lldb::TargetSP target_sp;
    if (exe_scope)
        target_sp = exe_scope->CalculateTarget();
    
    // TODO: figure out what to really do when we don't have a valid target.
    // Sometimes this will be ok to just use the host target triple (when we
    // evaluate say "2+3", but other expressions like breakpoint conditions
    // and other things that _are_ target specific really shouldn't just be
    // using the host triple. This needs to be fixed in a better way.
    if (target_sp && target_sp->GetArchitecture().IsValid())
    {
        std::string triple = target_sp->GetArchitecture().GetTriple().str();
        
        int dash_count = 0;
        for (size_t i = 0; i < triple.size(); ++i)
        {
            if (triple[i] == '-')
                dash_count++;
            if (dash_count == 3)
            {
                triple.resize(i);
                break;
            }
        }
        
        m_compiler->getTargetOpts().Triple = triple;
    }
    else
    {
        m_compiler->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();
    }
    
    if (m_compiler->getTargetOpts().Triple.find("ios") != std::string::npos)
        m_compiler->getTargetOpts().ABI = "apcs-gnu";
    
    m_compiler->createDiagnostics(0, 0);
    
    // Create the target instance.
    m_compiler->setTarget(TargetInfo::CreateTargetInfo(m_compiler->getDiagnostics(),
                                                       m_compiler->getTargetOpts()));
    
    assert (m_compiler->hasTarget());
    
    // 3. Set options.
    
    lldb::LanguageType language = expr.Language();
    
    switch (language)
    {
    case lldb::eLanguageTypeC:
        break;
    case lldb::eLanguageTypeObjC:
        m_compiler->getLangOpts().ObjC1 = true;
        m_compiler->getLangOpts().ObjC2 = true;
        break;
    case lldb::eLanguageTypeC_plus_plus:
        m_compiler->getLangOpts().CPlusPlus = true;
        m_compiler->getLangOpts().CPlusPlus0x = true;
        break;
    case lldb::eLanguageTypeObjC_plus_plus:
    default:
        m_compiler->getLangOpts().ObjC1 = true;
        m_compiler->getLangOpts().ObjC2 = true;
        m_compiler->getLangOpts().CPlusPlus = true;
        m_compiler->getLangOpts().CPlusPlus0x = true;
        break;
    }
    
    m_compiler->getLangOpts().Bool = true;
    m_compiler->getLangOpts().WChar = true;
    m_compiler->getLangOpts().DebuggerSupport = true; // Features specifically for debugger clients
    if (expr.DesiredResultType() == ClangExpression::eResultTypeId)
        m_compiler->getLangOpts().DebuggerCastResultToId = true;
    
    // Spell checking is a nice feature, but it ends up completing a
    // lot of types that we didn't strictly speaking need to complete.
    // As a result, we spend a long time parsing and importing debug
    // information.
    m_compiler->getLangOpts().SpellChecking = false; 
    
    lldb::ProcessSP process_sp;
    if (exe_scope)
        process_sp = exe_scope->CalculateProcess();

    if (process_sp && m_compiler->getLangOpts().ObjC1)
    {
        if (process_sp->GetObjCLanguageRuntime())
        {
            if (process_sp->GetObjCLanguageRuntime()->GetRuntimeVersion() == eAppleObjC_V2)
                m_compiler->getLangOpts().ObjCRuntime.set(ObjCRuntime::MacOSX, VersionTuple(10, 7));
            else
                m_compiler->getLangOpts().ObjCRuntime.set(ObjCRuntime::FragileMacOSX, VersionTuple(10, 7));
            
            if (process_sp->GetObjCLanguageRuntime()->HasNewLiteralsAndIndexing())
                m_compiler->getLangOpts().DebuggerObjCLiteral = true;
        }
    }

    m_compiler->getLangOpts().ThreadsafeStatics = false;
    m_compiler->getLangOpts().AccessControl = false; // Debuggers get universal access
    m_compiler->getLangOpts().DollarIdents = true; // $ indicates a persistent variable name
    
    // Set CodeGen options
    m_compiler->getCodeGenOpts().EmitDeclMetadata = true;
    m_compiler->getCodeGenOpts().InstrumentFunctions = false;
    
    // Disable some warnings.
    m_compiler->getDiagnosticOpts().Warnings.push_back("no-unused-value");
    
    // Inform the target of the language options
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    m_compiler->getTarget().setForcedLangOptions(m_compiler->getLangOpts());
    
    // 4. Set up the diagnostic buffer for reporting errors
    
    m_compiler->getDiagnostics().setClient(new clang::TextDiagnosticBuffer);
    
    // 5. Set up the source management objects inside the compiler
    
    clang::FileSystemOptions file_system_options;
    m_file_manager.reset(new clang::FileManager(file_system_options));
    
    if (!m_compiler->hasSourceManager())
        m_compiler->createSourceManager(*m_file_manager.get());
    
    m_compiler->createFileManager();
    m_compiler->createPreprocessor();
    
    // 6. Most of this we get from the CompilerInstance, but we 
    // also want to give the context an ExternalASTSource.
    m_selector_table.reset(new SelectorTable());
    m_builtin_context.reset(new Builtin::Context());
    
    std::auto_ptr<clang::ASTContext> ast_context(new ASTContext(m_compiler->getLangOpts(),
                                                                m_compiler->getSourceManager(),
                                                                &m_compiler->getTarget(),
                                                                m_compiler->getPreprocessor().getIdentifierTable(),
                                                                *m_selector_table.get(),
                                                                *m_builtin_context.get(),
                                                                0));
    
    ClangExpressionDeclMap *decl_map = m_expr.DeclMap();
    
    if (decl_map)
    {
        llvm::OwningPtr<clang::ExternalASTSource> ast_source(decl_map->CreateProxy());
        decl_map->InstallASTContext(ast_context.get());
        ast_context->setExternalSource(ast_source);
    }
    
    m_compiler->setASTContext(ast_context.release());
    
    std::string module_name("$__lldb_module");

    m_llvm_context.reset(new LLVMContext());
    m_code_generator.reset(CreateLLVMCodeGen(m_compiler->getDiagnostics(),
                                             module_name,
                                             m_compiler->getCodeGenOpts(),
                                             *m_llvm_context));
}

ClangExpressionParser::~ClangExpressionParser()
{
}

unsigned
ClangExpressionParser::Parse (Stream &stream)
{
    TextDiagnosticBuffer *diag_buf = static_cast<TextDiagnosticBuffer*>(m_compiler->getDiagnostics().getClient());
        
    diag_buf->FlushDiagnostics (m_compiler->getDiagnostics());
    
    MemoryBuffer *memory_buffer = MemoryBuffer::getMemBufferCopy(m_expr.Text(), __FUNCTION__);
    m_compiler->getSourceManager().createMainFileIDForMemBuffer (memory_buffer);
    
    diag_buf->BeginSourceFile(m_compiler->getLangOpts(), &m_compiler->getPreprocessor());
    
    ASTConsumer *ast_transformer = m_expr.ASTTransformer(m_code_generator.get());
    
    if (ast_transformer)
        ParseAST(m_compiler->getPreprocessor(), ast_transformer, m_compiler->getASTContext());
    else 
        ParseAST(m_compiler->getPreprocessor(), m_code_generator.get(), m_compiler->getASTContext());    
    
    diag_buf->EndSourceFile();
        
    TextDiagnosticBuffer::const_iterator diag_iterator;
    
    int num_errors = 0;
    
    for (diag_iterator = diag_buf->warn_begin();
         diag_iterator != diag_buf->warn_end();
         ++diag_iterator)
        stream.Printf("warning: %s\n", (*diag_iterator).second.c_str());
    
    num_errors = 0;
    
    for (diag_iterator = diag_buf->err_begin();
         diag_iterator != diag_buf->err_end();
         ++diag_iterator)
    {
        num_errors++;
        stream.Printf("error: %s\n", (*diag_iterator).second.c_str());
    }
    
    for (diag_iterator = diag_buf->note_begin();
         diag_iterator != diag_buf->note_end();
         ++diag_iterator)
        stream.Printf("note: %s\n", (*diag_iterator).second.c_str());
    
    if (!num_errors)
    {
        if (m_expr.DeclMap() && !m_expr.DeclMap()->ResolveUnknownTypes())
        {
            stream.Printf("error: Couldn't infer the type of a variable\n");
            num_errors++;
        }
    }
    
    return num_errors;
}

static bool FindFunctionInModule (std::string &mangled_name,
                                  llvm::Module *module,
                                  const char *orig_name)
{
    for (llvm::Module::iterator fi = module->getFunctionList().begin(), fe = module->getFunctionList().end();
         fi != fe;
         ++fi)
    {        
        if (fi->getName().str().find(orig_name) != std::string::npos)
        {
            mangled_name = fi->getName().str();
            return true;
        }
    }
    
    return false;
}

Error
ClangExpressionParser::PrepareForExecution (lldb::addr_t &func_allocation_addr, 
                                            lldb::addr_t &func_addr, 
                                            lldb::addr_t &func_end, 
                                            ExecutionContext &exe_ctx,
                                            IRForTarget::StaticDataAllocator *data_allocator,
                                            bool &evaluated_statically,
                                            lldb::ClangExpressionVariableSP &const_result,
                                            ExecutionPolicy execution_policy)
{
    func_allocation_addr = LLDB_INVALID_ADDRESS;
	func_addr = LLDB_INVALID_ADDRESS;
	func_end = LLDB_INVALID_ADDRESS;
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    std::auto_ptr<llvm::ExecutionEngine> execution_engine_ap;
    
    Error err;
    
    std::auto_ptr<llvm::Module> module_ap (m_code_generator->ReleaseModule());

    if (!module_ap.get())
    {
        err.SetErrorToGenericError();
        err.SetErrorString("IR doesn't contain a module");
        return err;
    }
    
    // Find the actual name of the function (it's often mangled somehow)
    
    std::string function_name;
    
    if (!FindFunctionInModule(function_name, module_ap.get(), m_expr.FunctionName()))
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find %s() in the module", m_expr.FunctionName());
        return err;
    }
    else
    {
        if (log)
            log->Printf("Found function %s for %s", function_name.c_str(), m_expr.FunctionName());
    }
    
    ClangExpressionDeclMap *decl_map = m_expr.DeclMap(); // result can be NULL
    
    if (decl_map)
    {
        Stream *error_stream = NULL;
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
            error_stream = &target->GetDebugger().GetErrorStream();
    
        IRForTarget ir_for_target(decl_map,
                                  m_expr.NeedsVariableResolution(),
                                  execution_policy,
                                  const_result,
                                  data_allocator,
                                  error_stream,
                                  function_name.c_str());
        
        bool ir_can_run = ir_for_target.runOnModule(*module_ap);
        
        Error &interpreter_error(ir_for_target.getInterpreterError());
        
        if (execution_policy != eExecutionPolicyAlways && interpreter_error.Success())
        {
            if (const_result)
                const_result->TransferAddress();
            evaluated_statically = true;
            err.Clear();
            return err;
        }
        
        Process *process = exe_ctx.GetProcessPtr();

        if (!process || execution_policy == eExecutionPolicyNever)
        {
            err.SetErrorToGenericError();
            if (execution_policy == eExecutionPolicyAlways)
                err.SetErrorString("Execution needed to run in the target, but the target can't be run");
            else
                err.SetErrorStringWithFormat("Interpreting the expression locally failed: %s", interpreter_error.AsCString());

            return err;
        }
        else if (!ir_can_run)
        {
            err.SetErrorToGenericError();
            err.SetErrorString("The expression could not be prepared to run in the target");
            
            return err;
        }
        
        if (execution_policy != eExecutionPolicyNever &&
            m_expr.NeedsValidation() && 
            process)
        {
            if (!process->GetDynamicCheckers())
            {                
                DynamicCheckerFunctions *dynamic_checkers = new DynamicCheckerFunctions();
                
                StreamString install_errors;
                
                if (!dynamic_checkers->Install(install_errors, exe_ctx))
                {
                    if (install_errors.GetString().empty())
                        err.SetErrorString ("couldn't install checkers, unknown error");
                    else
                        err.SetErrorString (install_errors.GetString().c_str());
                    
                    return err;
                }
                
                process->SetDynamicCheckers(dynamic_checkers);
                
                if (log)
                    log->Printf("== [ClangUserExpression::Evaluate] Finished installing dynamic checkers ==");
            }
            
            IRDynamicChecks ir_dynamic_checks(*process->GetDynamicCheckers(), function_name.c_str());
        
            if (!ir_dynamic_checks.runOnModule(*module_ap))
            {
                err.SetErrorToGenericError();
                err.SetErrorString("Couldn't add dynamic checks to the expression");
                return err;
            }
        }
    }
    
    // llvm will own this pointer when llvm::ExecutionEngine::createJIT is called 
    // below so we don't need to free it.
    RecordingMemoryManager *jit_memory_manager = new RecordingMemoryManager();
    
    std::string error_string;

    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        module_ap->print(oss, NULL);
        
        oss.flush();
        
        log->Printf ("Module being sent to JIT: \n%s", s.c_str());
    }
    llvm::Triple triple(module_ap->getTargetTriple());
    llvm::Function *function = module_ap->getFunction (function_name.c_str());
    EngineBuilder builder(module_ap.release());
    builder.setEngineKind(EngineKind::JIT)
        .setErrorStr(&error_string)
        .setRelocationModel(llvm::Reloc::PIC_)
        .setJITMemoryManager(jit_memory_manager)
        .setOptLevel(CodeGenOpt::Less)
        .setAllocateGVsWithCode(true)
        .setCodeModel(CodeModel::Small)
        .setUseMCJIT(true);
    
    StringRef mArch;
    StringRef mCPU;
    SmallVector<std::string, 0> mAttrs;
    
    TargetMachine *target_machine = builder.selectTarget(triple,
                                                         mArch,
                                                         mCPU,
                                                         mAttrs);
    
    execution_engine_ap.reset(builder.create(target_machine));
        
    if (!execution_engine_ap.get())
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't JIT the function: %s", error_string.c_str());
        return err;
    }
    
    execution_engine_ap->DisableLazyCompilation();
    
    
    // We don't actually need the function pointer here, this just forces it to get resolved.
    
    void *fun_ptr = execution_engine_ap->getPointerToFunction(function);
        
    // Errors usually cause failures in the JIT, but if we're lucky we get here.
    
    if (!function)
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find '%s' in the JITted module", function_name.c_str());
        return err;
    }
    
    if (!fun_ptr)
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("'%s' was in the JITted module but wasn't lowered", function_name.c_str());
        return err;
    }
    
    m_jitted_functions.push_back (ClangExpressionParser::JittedFunction(function_name.c_str(), (lldb::addr_t)fun_ptr));
    

    Process *process = exe_ctx.GetProcessPtr();
    if (process == NULL)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Couldn't write the JIT compiled code into the target because there is no target");
        return err;
    }
        
    jit_memory_manager->CommitAllocations(*process);
    jit_memory_manager->ReportAllocations(*execution_engine_ap);
    jit_memory_manager->WriteData(*process);
    
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();
    
    for (pos = m_jitted_functions.begin(); pos != end; pos++)
    {
        (*pos).m_remote_addr = jit_memory_manager->GetRemoteAddressForLocal ((*pos).m_local_addr);
    
        if (!(*pos).m_name.compare(function_name.c_str()))
        {
            RecordingMemoryManager::AddrRange func_range = jit_memory_manager->GetRemoteRangeForLocal((*pos).m_local_addr);
            func_end = func_range.first + func_range.second;
            func_addr = (*pos).m_remote_addr;
        }
    }
    
    if (log)
    {
        log->Printf("Code can be run in the target.");
        
        StreamString disassembly_stream;
        
        Error err = DisassembleFunction(disassembly_stream, exe_ctx, jit_memory_manager);
        
        if (!err.Success())
        {
            log->Printf("Couldn't disassemble function : %s", err.AsCString("unknown error"));
        }
        else
        {
            log->Printf("Function disassembly:\n%s", disassembly_stream.GetData());
        }
    }
    
    err.Clear();
    return err;
}

Error
ClangExpressionParser::DisassembleFunction (Stream &stream, ExecutionContext &exe_ctx, RecordingMemoryManager *jit_memory_manager)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    const char *name = m_expr.FunctionName();
    
    Error ret;
    
    ret.Clear();
    
    lldb::addr_t func_local_addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t func_remote_addr = LLDB_INVALID_ADDRESS;
    
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();
    
    for (pos = m_jitted_functions.begin(); pos < end; pos++)
    {
        if (strstr(pos->m_name.c_str(), name))
        {
            func_local_addr = pos->m_local_addr;
            func_remote_addr = pos->m_remote_addr;
        }
    }
    
    if (func_local_addr == LLDB_INVALID_ADDRESS)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find function %s for disassembly", name);
        return ret;
    }
    
    if (log)
        log->Printf("Found function, has local address 0x%llx and remote address 0x%llx", (uint64_t)func_local_addr, (uint64_t)func_remote_addr);
    
    std::pair <lldb::addr_t, lldb::addr_t> func_range;
    
    func_range = jit_memory_manager->GetRemoteRangeForLocal(func_local_addr);
    
    if (func_range.first == 0 && func_range.second == 0)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find code range for function %s", name);
        return ret;
    }
    
    if (log)
        log->Printf("Function's code range is [0x%llx+0x%llx]", func_range.first, func_range.second);
    
    Target *target = exe_ctx.GetTargetPtr();
    if (!target)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the target");
        return ret;
    }
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(func_range.second, 0));
    
    Process *process = exe_ctx.GetProcessPtr();
    Error err;
    process->ReadMemory(func_remote_addr, buffer_sp->GetBytes(), buffer_sp->GetByteSize(), err);
    
    if (!err.Success())
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't read from process: %s", err.AsCString("unknown error"));
        return ret;
    }
    
    ArchSpec arch(target->GetArchitecture());
    
    lldb::DisassemblerSP disassembler = Disassembler::FindPlugin(arch, NULL);
    
    if (!disassembler)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Unable to find disassembler plug-in for %s architecture.", arch.GetArchitectureName());
        return ret;
    }
    
    if (!process)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the process");
        return ret;
    }
    
    DataExtractor extractor(buffer_sp, 
                            process->GetByteOrder(),
                            target->GetArchitecture().GetAddressByteSize());
    
    if (log)
    {
        log->Printf("Function data has contents:");
        extractor.PutToLog (log.get(),
                            0,
                            extractor.GetByteSize(),
                            func_remote_addr,
                            16,
                            DataExtractor::TypeUInt8);
    }
    
    disassembler->DecodeInstructions (Address (func_remote_addr), extractor, 0, UINT32_MAX, false);
    
    InstructionList &instruction_list = disassembler->GetInstructionList();
    const uint32_t max_opcode_byte_size = instruction_list.GetMaxOpcocdeByteSize();
    for (uint32_t instruction_index = 0, num_instructions = instruction_list.GetSize(); 
         instruction_index < num_instructions; 
         ++instruction_index)
    {
        Instruction *instruction = instruction_list.GetInstructionAtIndex(instruction_index).get();
        instruction->Dump (&stream,
                           max_opcode_byte_size,
                           true,
                           true,
                           &exe_ctx);
        stream.PutChar('\n');
    }
    
    return ret;
}
