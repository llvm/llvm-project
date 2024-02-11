#include "CallGraph.h"

std::map<std::string, std::set<std::string>>
    GenWholeProgramCallGraphVisitor::callGraph;

std::map<std::string, NamedLocation *>
    GenWholeProgramCallGraphVisitor::infoOfFunction;

std::string
GenWholeProgramCallGraphVisitor::getMangledName(const FunctionDecl *decl) {
    auto mangleContext = Context->createMangleContext();

    if (!mangleContext->shouldMangleDeclName(decl)) {
        return decl->getNameInfo().getName().getAsString();
    }

    std::string mangledName;
    llvm::raw_string_ostream ostream(mangledName);

    mangleContext->mangleName(decl, ostream);

    ostream.flush();

    delete mangleContext;

    return mangledName;
};

bool GenWholeProgramCallGraphVisitor::VisitFunctionDecl(FunctionDecl *D) {

    FullSourceLoc FullLocation =
        D->getASTContext().getFullLoc(D->getBeginLoc());
    if (FullLocation.isInvalid() || !FullLocation.hasManager())
        return true;
    const FileEntry *fileEntry = FullLocation.getFileEntry();
    if (!fileEntry)
        return true;
    StringRef _file = fileEntry->tryGetRealPathName();
    requireTrue(!_file.empty());
    std::string file(_file);

    /**
     * 目前，跳过不在当前文件的函数
     *
     * 被跳过的主要包含库函数。
     * 但如果 .h 中包含了函数定义(例如 inline 和 template
     * 函数)，也会被省略。
     *
     * See: Is it a good practice to place C++ definitions in header files?
     *      https://stackoverflow.com/a/583271
     */
    if (filePath != file)
        return true;

    if (!D->isThisDeclarationADefinition())
        return true;

    std::string fullSignature = getFullSignature(D);
    // declaration already processed
    // 由于 include，可能导致重复定义？
    if (infoOfFunction.find(fullSignature) != infoOfFunction.end())
        return true;

    infoOfFunction[fullSignature] =
        new NamedLocation(file, FullLocation.getLineNumber(),
                          FullLocation.getColumnNumber(), fullSignature);

    CallGraph CG;
    CG.addToCallGraph(D);
    // CG.dump();

    CallGraphNode *N = CG.getNode(D->getCanonicalDecl());
    requireTrue(N != nullptr, "N is null!");
    for (CallGraphNode::const_iterator CI = N->begin(), CE = N->end(); CI != CE;
         ++CI) {
        FunctionDecl *callee = CI->Callee->getDecl()->getAsFunction();
        requireTrue(callee != nullptr, "callee is null!");
        callGraph[fullSignature].insert(getFullSignature(callee));
    }

    return true;
}

bool GenWholeProgramCallGraphVisitor::VisitCXXRecordDecl(CXXRecordDecl *D) {
    // llvm::errs() << D->getQualifiedNameAsString() << "\n";
    return true;
}

void GenWholeProgramCallGraphConsumer::HandleTranslationUnit(
    clang::ASTContext &Context) {
    TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
    // TUD->dump();
    // CallGraph CG;
    // CG.addToCallGraph(TUD);
    // CG.dump();
    Visitor.TraverseDecl(TUD);
}

std::unique_ptr<clang::ASTConsumer>
GenWholeProgramCallGraphAction::CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    fs::path filePath = fs::canonical(BUILD_PATH / std::string(InFile));
    llvm::outs() << "CreateASTConsumer in file: " << filePath << "\n";
    return std::make_unique<GenWholeProgramCallGraphConsumer>(
        &Compiler.getASTContext(), filePath);
}