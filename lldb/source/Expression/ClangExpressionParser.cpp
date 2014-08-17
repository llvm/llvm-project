//===-- ClangExpressionParser.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Expression/ClangExpressionParser.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/IRDynamicChecks.h"
#include "lldb/Expression/IRInterpreter.h"
#include "lldb/Host/File.h"
#include "lldb/Symbol/SymbolVendor.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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
    SmallString<128> P(llvm::sys::fs::getMainExecutable(
        Argv0, (void *)(intptr_t) GetBuiltinIncludePath));

    if (!P.empty()) {
        llvm::sys::path::remove_filename(P); // Remove /clang from foo/bin/clang
        llvm::sys::path::remove_filename(P); // Remove /bin   from foo/bin

        // Get foo/lib/clang/<version>/include
        llvm::sys::path::append(P, "lib", "clang", CLANG_VERSION_STRING,
                                "include");
    }

    return P.str();
}

//===----------------------------------------------------------------------===//
// Implementation of ClangExpressionParser
//===----------------------------------------------------------------------===//

ClangExpressionParser::ClangExpressionParser (ExecutionContextScope *exe_scope,
                                              ClangExpression &expr,
                                              bool generate_debug_info) :
    m_expr (expr),
    m_compiler (),
    m_code_generator ()
{
    // Initialize targets first, so that --version shows registered targets.
    static struct InitializeLLVM {
        InitializeLLVM() {
            llvm::InitializeAllTargets();
            llvm::InitializeAllAsmPrinters();
            llvm::InitializeAllTargetMCs();
            llvm::InitializeAllDisassemblers();
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
        m_compiler->getTargetOpts().Triple = triple;
    }
    else
    {
        m_compiler->getTargetOpts().Triple = llvm::sys::getDefaultTargetTriple();
    }

    if (target_sp->GetArchitecture().GetMachine() == llvm::Triple::x86 ||
        target_sp->GetArchitecture().GetMachine() == llvm::Triple::x86_64)
    {
        m_compiler->getTargetOpts().Features.push_back("+sse");
        m_compiler->getTargetOpts().Features.push_back("+sse2");
    }

    // Any arm32 iOS environment, but not on arm64
    if (m_compiler->getTargetOpts().Triple.find("arm64") == std::string::npos &&
        m_compiler->getTargetOpts().Triple.find("arm") != std::string::npos &&
        m_compiler->getTargetOpts().Triple.find("ios") != std::string::npos)
    {
        m_compiler->getTargetOpts().ABI = "apcs-gnu";
    }

    m_compiler->createDiagnostics();

    // Create the target instance.
    m_compiler->setTarget(TargetInfo::CreateTargetInfo(
        m_compiler->getDiagnostics(), m_compiler->getInvocation().TargetOpts));

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
        m_compiler->getLangOpts().CPlusPlus11 = true;
        break;
    case lldb::eLanguageTypeObjC_plus_plus:
    default:
        m_compiler->getLangOpts().ObjC1 = true;
        m_compiler->getLangOpts().ObjC2 = true;
        m_compiler->getLangOpts().CPlusPlus = true;
        m_compiler->getLangOpts().CPlusPlus11 = true;
        break;
    }

    m_compiler->getLangOpts().Bool = true;
    m_compiler->getLangOpts().WChar = true;
    m_compiler->getLangOpts().Blocks = true;
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
    m_compiler->getCodeGenOpts().DisableFPElim = true;
    m_compiler->getCodeGenOpts().OmitLeafFramePointer = false;
    if (generate_debug_info)
        m_compiler->getCodeGenOpts().setDebugInfo(CodeGenOptions::FullDebugInfo);
    else
        m_compiler->getCodeGenOpts().setDebugInfo(CodeGenOptions::NoDebugInfo);

    // Disable some warnings.
    m_compiler->getDiagnostics().setSeverityForGroup(
        clang::diag::Flavor::WarningOrError, "unused-value",
        clang::diag::Severity::Ignored, SourceLocation());
    m_compiler->getDiagnostics().setSeverityForGroup(
        clang::diag::Flavor::WarningOrError, "odr",
        clang::diag::Severity::Ignored, SourceLocation());

    // Inform the target of the language options
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    m_compiler->getTarget().adjust(m_compiler->getLangOpts());

    // 4. Set up the diagnostic buffer for reporting errors

    m_compiler->getDiagnostics().setClient(new clang::TextDiagnosticBuffer);

    // 5. Set up the source management objects inside the compiler

    clang::FileSystemOptions file_system_options;
    m_file_manager.reset(new clang::FileManager(file_system_options));

    if (!m_compiler->hasSourceManager())
        m_compiler->createSourceManager(*m_file_manager.get());

    m_compiler->createFileManager();
    m_compiler->createPreprocessor(TU_Complete);

    // 6. Most of this we get from the CompilerInstance, but we
    // also want to give the context an ExternalASTSource.
    m_selector_table.reset(new SelectorTable());
    m_builtin_context.reset(new Builtin::Context());

    std::unique_ptr<clang::ASTContext> ast_context(new ASTContext(m_compiler->getLangOpts(),
                                                                 m_compiler->getSourceManager(),
                                                                 m_compiler->getPreprocessor().getIdentifierTable(),
                                                                 *m_selector_table.get(),
                                                                 *m_builtin_context.get()));
    ast_context->InitBuiltinTypes(m_compiler->getTarget());

    ClangExpressionDeclMap *decl_map = m_expr.DeclMap();

    if (decl_map)
    {
        llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> ast_source(decl_map->CreateProxy());
        decl_map->InstallASTContext(ast_context.get());
        ast_context->setExternalSource(ast_source);
    }

    m_compiler->setASTContext(ast_context.release());

    std::string module_name("$__lldb_module");

    m_llvm_context.reset(new LLVMContext());
    m_code_generator.reset(CreateLLVMCodeGen(m_compiler->getDiagnostics(),
                                             module_name,
                                             m_compiler->getCodeGenOpts(),
                                             m_compiler->getTargetOpts(),
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

    const char *expr_text = m_expr.Text();

    clang::SourceManager &SourceMgr = m_compiler->getSourceManager();
    bool created_main_file = false;
    if (m_compiler->getCodeGenOpts().getDebugInfo() == CodeGenOptions::FullDebugInfo)
    {
        std::string temp_source_path;

        FileSpec tmpdir_file_spec;
        if (Host::GetLLDBPath (ePathTypeLLDBTempSystemDir, tmpdir_file_spec))
        {
            tmpdir_file_spec.GetFilename().SetCString("expr.XXXXXX");
            temp_source_path = std::move(tmpdir_file_spec.GetPath());
        }
        else
        {
            temp_source_path = "/tmp/expr.XXXXXX";
        }

        if (mktemp(&temp_source_path[0]))
        {
            lldb_private::File file (temp_source_path.c_str(),
                                     File::eOpenOptionWrite | File::eOpenOptionCanCreateNewOnly,
                                     lldb::eFilePermissionsFileDefault);
            const size_t expr_text_len = strlen(expr_text);
            size_t bytes_written = expr_text_len;
            if (file.Write(expr_text, bytes_written).Success())
            {
                if (bytes_written == expr_text_len)
                {
                    file.Close();
                    SourceMgr.setMainFileID(SourceMgr.createFileID(
                        m_file_manager->getFile(temp_source_path),
                        SourceLocation(), SrcMgr::C_User));
                    created_main_file = true;
                }
            }
        }
    }

    if (!created_main_file)
    {
        MemoryBuffer *memory_buffer = MemoryBuffer::getMemBufferCopy(expr_text, __FUNCTION__);
        SourceMgr.setMainFileID(SourceMgr.createFileID(memory_buffer));
    }

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

static bool FindFunctionInModule (ConstString &mangled_name,
                                  llvm::Module *module,
                                  const char *orig_name)
{
    for (llvm::Module::iterator fi = module->getFunctionList().begin(), fe = module->getFunctionList().end();
         fi != fe;
         ++fi)
    {
        if (fi->getName().str().find(orig_name) != std::string::npos)
        {
            mangled_name.SetCString(fi->getName().str().c_str());
            return true;
        }
    }

    return false;
}

Error
ClangExpressionParser::PrepareForExecution (lldb::addr_t &func_addr,
                                            lldb::addr_t &func_end,
                                            std::shared_ptr<IRExecutionUnit> &execution_unit_sp,
                                            ExecutionContext &exe_ctx,
                                            bool &can_interpret,
                                            ExecutionPolicy execution_policy)
{
	func_addr = LLDB_INVALID_ADDRESS;
	func_end = LLDB_INVALID_ADDRESS;
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    Error err;

    std::unique_ptr<llvm::Module> llvm_module_ap (m_code_generator->ReleaseModule());

    if (!llvm_module_ap.get())
    {
        err.SetErrorToGenericError();
        err.SetErrorString("IR doesn't contain a module");
        return err;
    }

    // Find the actual name of the function (it's often mangled somehow)

    ConstString function_name;

    if (!FindFunctionInModule(function_name, llvm_module_ap.get(), m_expr.FunctionName()))
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find %s() in the module", m_expr.FunctionName());
        return err;
    }
    else
    {
        if (log)
            log->Printf("Found function %s for %s", function_name.AsCString(), m_expr.FunctionName());
    }

    execution_unit_sp.reset(new IRExecutionUnit (m_llvm_context, // handed off here
                                                 llvm_module_ap, // handed off here
                                                 function_name,
                                                 exe_ctx.GetTargetSP(),
                                                 m_compiler->getTargetOpts().Features));

    ClangExpressionDeclMap *decl_map = m_expr.DeclMap(); // result can be NULL

    if (decl_map)
    {
        Stream *error_stream = NULL;
        Target *target = exe_ctx.GetTargetPtr();
        if (target)
            error_stream = target->GetDebugger().GetErrorFile().get();

        IRForTarget ir_for_target(decl_map,
                                  m_expr.NeedsVariableResolution(),
                                  *execution_unit_sp,
                                  error_stream,
                                  function_name.AsCString());

        bool ir_can_run = ir_for_target.runOnModule(*execution_unit_sp->GetModule());

        Error interpret_error;

        can_interpret = IRInterpreter::CanInterpret(*execution_unit_sp->GetModule(), *execution_unit_sp->GetFunction(), interpret_error);

        Process *process = exe_ctx.GetProcessPtr();

        if (!ir_can_run)
        {
            err.SetErrorString("The expression could not be prepared to run in the target");
            return err;
        }

        if (!can_interpret && execution_policy == eExecutionPolicyNever)
        {
            err.SetErrorStringWithFormat("Can't run the expression locally: %s", interpret_error.AsCString());
            return err;
        }

        if (!process && execution_policy == eExecutionPolicyAlways)
        {
            err.SetErrorString("Expression needed to run in the target, but the target can't be run");
            return err;
        }

        if (execution_policy == eExecutionPolicyAlways || !can_interpret)
        {
            if (m_expr.NeedsValidation() && process)
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

                IRDynamicChecks ir_dynamic_checks(*process->GetDynamicCheckers(), function_name.AsCString());

                if (!ir_dynamic_checks.runOnModule(*execution_unit_sp->GetModule()))
                {
                    err.SetErrorToGenericError();
                    err.SetErrorString("Couldn't add dynamic checks to the expression");
                    return err;
                }
            }

            execution_unit_sp->GetRunnableInfo(err, func_addr, func_end);
        }
    }
    else
    {
        execution_unit_sp->GetRunnableInfo(err, func_addr, func_end);
    }

    return err;
}

bool
ClangExpressionParser::GetGenerateDebugInfo () const
{
    if (m_compiler)
        return m_compiler->getCodeGenOpts().getDebugInfo() == CodeGenOptions::FullDebugInfo;
    return false;
}
