#include "GenICFG.h"
#include "ICFG.h"
#include "utils.h"

std::string GenICFGVisitor::getMangledName(const FunctionDecl *decl) {
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

bool GenICFGVisitor::VisitFunctionDecl(FunctionDecl *D) {
    std::unique_ptr<Location> pLoc =
        Location::fromSourceLocation(D->getASTContext(), D->getBeginLoc());
    if (!pLoc)
        return true;

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
    if (filePath != pLoc->file)
        return true;

    if (!D->isThisDeclarationADefinition())
        return true;

    if (!D->getBody())
        return true;

    std::string fullSignature = getFullSignature(D);

    auto &functionCnt = Global.functionCnt;
    auto &idOfFunction = Global.idOfFunction;
    auto &functionLocations = Global.functionLocations;

    // declaration already processed
    // 由于 include，可能导致重复定义？
    if (idOfFunction.find(fullSignature) != idOfFunction.end())
        return true;

    NamedLocation *loc = new NamedLocation(*pLoc, fullSignature);

    idOfFunction[fullSignature] = functionCnt++;
    functionLocations.push_back(loc);

    CallGraph CG;
    CG.addToCallGraph(D);
    // CG.dump();

    CallGraphNode *N = CG.getNode(D->getCanonicalDecl());
    requireTrue(N != nullptr, "N is null!");
    for (CallGraphNode::const_iterator CI = N->begin(), CE = N->end(); CI != CE;
         ++CI) {
        FunctionDecl *callee = CI->Callee->getDecl()->getAsFunction();
        requireTrue(callee != nullptr, "callee is null!");
        Global.callGraph[fullSignature].insert(getFullSignature(callee));
    }

    std::unique_ptr<CFG> cfg = CFG::buildCFG(
        D, D->getBody(), &D->getASTContext(), CFG::BuildOptions());
    Global.icfg.addFunction(Global.getIdOfFunction(fullSignature), *cfg);

    return true;
}

bool GenICFGVisitor::VisitCXXRecordDecl(CXXRecordDecl *D) {
    // llvm::errs() << D->getQualifiedNameAsString() << "\n";
    return true;
}

void GenICFGConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
    // TUD->dump();
    // CallGraph CG;
    // CG.addToCallGraph(TUD);
    // CG.dump();
    Visitor.TraverseDecl(TUD);
}

std::unique_ptr<clang::ASTConsumer>
GenICFGAction::CreateASTConsumer(clang::CompilerInstance &Compiler,
                                 llvm::StringRef InFile) {

    static const int total = Global.cb->getAllFiles().size();
    static int fileCnt = 0;
    fileCnt++;

    SourceManager &sm = Compiler.getSourceManager();
    const FileEntry *fileEntry = sm.getFileEntryForID(sm.getMainFileID());
    std::string filePath(fileEntry->tryGetRealPathName());
    llvm::errs() << "[" << fileCnt << "/" << total << "] " << filePath << "\n";

    return std::make_unique<GenICFGConsumer>(&Compiler.getASTContext(),
                                             filePath);
}