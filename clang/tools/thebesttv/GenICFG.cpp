#include "GenICFG.h"
#include "DumpPath.h"
#include "GenAST.h"
#include "ICFG.h"
#include "utils.h"

#include "lib/BS_thread_pool.hpp"

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

    // 记录 .h 文件的 AST 对应 cc.json 中哪个文件
    auto &sourceForFile = Global.icfg.sourceForFile;
    if (sourceForFile.find(pLoc->file) == sourceForFile.end()) {
        sourceForFile[pLoc->file] = filePath;
    }

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
    auto &fileAndIdOfFunction = Global.fileAndIdOfFunction;
    auto &functionLocations = Global.functionLocations;

    // declaration already processed
    auto it = fileAndIdOfFunction.find(fullSignature);
    if (it != fileAndIdOfFunction.end()) { // 首先看函数签名是否已经见过
        auto &fidOfFile = it->second;
        // 然后判断函数定义所在的文件是不是也见过了
        if (fidOfFile.find(pLoc->file) != fidOfFile.end()) {
            // logger.warn("Skipping func {} in {}:{}:{}", fullSignature,
            //             pLoc->file, pLoc->line, pLoc->column);
            return true;
        }
    }

    std::unique_ptr<CFG> cfg = CFG::buildCFG(
        D, D->getBody(), &D->getASTContext(), CFG::BuildOptions());
    if (!cfg) {
        logger.error("Unable to create CFG for function {} in {}:{}:{}",
                     fullSignature, pLoc->file, pLoc->line, pLoc->column);
        return true;
    }

    fileAndIdOfFunction[fullSignature][pLoc->file] = functionCnt++;
    functionLocations.emplace_back(*pLoc, fullSignature);

    CallGraph CG;
    CG.addToCallGraph(D);
    // CG.dump();

    CallGraphNode *N = CG.getNode(D->getCanonicalDecl());
    if (N == nullptr) {
        /**
         * 不知道为什么，CG还是有可能只有一个root节点。
         *
         * 样例 linux，版本 6.7.5.arch1-1
         * linux/src/linux-6.7.5/include/linux/bsearch.h 的函数
         * __inline_bsearch()
         */
        requireTrue(CG.size() == 1, "Empty call graph! (only root node)");
    } else {
        for (CallGraphNode::const_iterator CI = N->begin(), CE = N->end();
             CI != CE; ++CI) {
            FunctionDecl *callee = CI->Callee->getDecl()->getAsFunction();
            requireTrue(callee != nullptr, "callee is null!");
            Global.callGraph[fullSignature].insert(getFullSignature(callee));
        }
    }

    Global.icfg.addFunction(Global.getIdOfFunction(fullSignature, pLoc->file),
                            *cfg);

    NpeSourceVisitor(Context).TraverseDecl(D);

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

bool isPointerType(Expr *E) {
    auto type = E->getType().getTypePtrOrNull();
    return type && type->isAnyPointerType();
}

std::optional<typename std::set<ordered_json>::iterator>
NpeSourceVisitor::saveNpeSuspectedSources(const SourceRange &range) {
    ordered_json loc;
    // something wrong with location
    if (!saveLocationInfo(*Context, range, loc))
        return std::nullopt;
    // source outside current project
    const std::string &file = loc["file"];
    if (!Global.isUnderProject(file))
        return std::nullopt;

    return reservoirSamplingAddElement(Global.npeSuspectedSources, loc, 100000);
}

bool NpeSourceVisitor::VisitVarDecl(VarDecl *D) {
    // 加入 NPE 可疑的 source 中

    // must be declared within a function
    if (!D->getParentFunctionOrMethod())
        return true;

    Expr *init = D->getInit();
    if (!init || !isPointerType(init))
        return true;

    saveNpeSuspectedSources(D->getSourceRange());

    return true;
}

bool NpeSourceVisitor::VisitBinaryOperator(BinaryOperator *S) {
    // 加入 NPE 可疑的 source 中

    // equivalent to: !(S->isAssignmentOp() && !S->isCompoundAssignmentOp())
    // i.e. is assignment & not compount assignment
    if (S->getOpcode() != BO_Assign)
        return true;

    if (!isPointerType(S))
        return true;

    saveNpeSuspectedSources(S->getSourceRange());

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
    logger.info("[{}/{}] {}", cnt, total, filePath);

    return std::make_unique<GenICFGConsumer>(&Compiler.getASTContext(),
                                             filePath);
}

std::mutex icfgMtx;
bool updateICFGWithASTDump(const std::string &file) {
    auto AST = getASTOfFile(file);
    if (AST) {
        icfgMtx.lock();
        auto &Context = AST->getASTContext();
        GenICFGVisitor visitor(&Context, file);
        visitor.TraverseDecl(Context.getTranslationUnitDecl());
        icfgMtx.unlock();
        llvm::sys::fs::remove(getASTDumpFile(file));
        return true;
    }
    return false;
}

void generateICFG(const CompilationDatabase &cb) {
    logger.info("--- Generating whole program call graph ---");

    auto allCmds = cb.getAllCompileCommands();
    ProgressBar bar("Gen ICFG", allCmds.size());
    int badCnt = 0, goodCnt = 0;
    for (auto &cmd : allCmds) {
        int ret = generateASTDump(cmd);
        if (ret == 0) {
            bool result = updateICFGWithASTDump(cmd.Filename);
            if (result == true) {
                goodCnt++;
            } else {
                badCnt++;
            }
        } else {
            badCnt++;
        }
        bar.tick();
    }
    bar.done();

    logger.info("ICFG generation finished with {} success and {} failure",
                goodCnt, badCnt);
}

void generateICFGParallel(const CompilationDatabase &cb, int numThreads) {
    logger.info("--- Generating whole program call graph (parallel) ---");

    BS::thread_pool pool(numThreads);
    std::vector<std::future<int>> tasks;

    auto allCmds = cb.getAllCompileCommands();
    ProgressBar bar("Gen ICFG", allCmds.size());
    for (const auto &cmd : allCmds) {
        tasks.push_back(pool.submit_task([cmd, &bar] {
            int ret = generateASTDump(cmd);
            if (ret == 0) {
                bool result = updateICFGWithASTDump(cmd.Filename);
                requireTrue(result == true);
            }
            bar.tick();
            return ret;
        }));
    }

    int badCnt = 0, goodCnt = 0;
    for (auto &task : tasks) {
        int ret = task.get();
        ret == 0 ? goodCnt++ : badCnt++;
    }
    bar.done();

    logger.info("ICFG generation finished with {} success and {} failure",
                goodCnt, badCnt);
}
