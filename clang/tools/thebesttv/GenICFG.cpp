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
     * 跳过不在 Global.projectDirectory 中的函数。
     *
     * See: Is it a good practice to place C++ definitions in header files?
     *      https://stackoverflow.com/a/583271
     */
    if (!Global.isUnderProject(pLoc->file))
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

    idOfFunction[fullSignature] = functionCnt++;
    functionLocations.emplace_back(*pLoc, fullSignature);

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

    /*
    // traverse CFGBlocks
    for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
        CFGBlock &B = **BI;
        // traverse all stmts
        for (auto SI = B.begin(); SI != B.end(); ++SI) {
            CFGElement &E = *SI;
            if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                const Stmt *S = CS->getStmt();
                dumpStmt(*Context, S);
                for (const Stmt *child : S->children()) {
                    if (child) {
                        dumpStmt(*Context, child);
                    }
                }
            }
        }
    }
    */

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

    static const int total = Global.cb->getAllCompileCommands().size();
    static int cnt = 0;
    cnt++;

    SourceManager &sm = Compiler.getSourceManager();
    const FileEntry *fileEntry = sm.getFileEntryForID(sm.getMainFileID());
    std::string filePath(fileEntry->tryGetRealPathName());
    llvm::errs() << "[" << cnt << "/" << total << "] " << filePath << "\n";

    return std::make_unique<GenICFGConsumer>(&Compiler.getASTContext(),
                                             filePath);
}