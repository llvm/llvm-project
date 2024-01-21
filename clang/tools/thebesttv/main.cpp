#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include <filesystem>
#include <queue>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
namespace fs = std::filesystem;

/*****************************************************************
 * Global Variables
 *****************************************************************/

fs::path BUILD_PATH;

struct FunctionInfo;
std::map<std::string, std::set<const FunctionInfo *>> functionsInFile;

/*****************************************************************
 * Main body
 *****************************************************************/

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");

// class tbtASTDumpAction : public ASTFrontendAction {
// protected:
//   std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
//                                                  StringRef InFile) override;
// public:
//   void ExecuteAction() override {
//     llvm::errs() << "tbtASTDumpAction::ExecuteAction\n";
//     llvm::errs() << "  " << getCurrentFile() << "\n";
//     // get declarations in ast
//     ASTContext &Context = getCompilerInstance().getASTContext();
//     TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
//     for (Decl *D : TUD->decls()) {
//       llvm::errs() << "  " << D->getDeclKindName() << "\n";
//     }
//   }
// };

// std::unique_ptr<ASTConsumer>
// tbtASTDumpAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
//   const FrontendOptions &Opts = CI.getFrontendOpts();
//   return CreateASTDumper(nullptr /*Dump to stdout.*/, Opts.ASTDumpFilter,
//                          Opts.ASTDumpDecls, Opts.ASTDumpAll,
//                          Opts.ASTDumpLookups, Opts.ASTDumpDeclTypes,
//                          Opts.ASTDumpFormat);
// }

void requireTrue(bool condition, std::string message = "") {
    if (!condition) {
        llvm::errs() << "requireTrue failed: " << message << "\n";
        exit(1);
    }
}

class HasContextVisitor {
  protected:
    ASTContext *Context;

    std::string getLocation(const SourceLocation &loc) {
        PresumedLoc PLoc = Context->getSourceManager().getPresumedLoc(loc);
        if (PLoc.isInvalid())
            return "<invalid>";

        std::string filename = PLoc.getFilename();
        std::string line = std::to_string(PLoc.getLine());
        std::string column = std::to_string(PLoc.getColumn());
        return filename + ":" + line + ":" + column;
    }

    void printStmtLocation(const Stmt &s) {
        llvm::errs() << "    beg: " << getLocation(s.getBeginLoc()) << "\n";
        llvm::errs() << "    end: " << getLocation(s.getEndLoc()) << "\n";
    }

  public:
    explicit HasContextVisitor(ASTContext *Context) : Context(Context) {}
};

/**
 * Visit all DeclRefExprs and print their parents.
 */
class VarVisitor : public RecursiveASTVisitor<VarVisitor>,
                   public HasContextVisitor {
  public:
    explicit VarVisitor(ASTContext *Context) : HasContextVisitor(Context) {}

    void visitParents(const Stmt &base) {
        const Stmt *s = &base;
        llvm::errs() << "    parents:\n";
        while (true) {
            const auto &parents = Context->getParents(*s);
            requireTrue(parents.size() == 1, "parent size is not 1");

            const Stmt *parent = parents.begin()->get<Stmt>();
            requireTrue(parent != nullptr, "parent is null");

            llvm::errs() << "      " << parent->getStmtClassName() << "\n";
            if (isa<CompoundStmt>(parent)) {
                break;
            }

            s = parent;
        }
    }

    bool VisitStmt(Stmt *s) {
        // DeclRefExpr
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(s)) {
            llvm::errs() << "  DeclRefExpr: "
                         << dre->getDecl()->getQualifiedNameAsString() << "\n";
            printStmtLocation(*s);
            visitParents(*s);
        }
        return true;
    }
};

struct VarLocation {
    std::string file; // absolute path
    int line;
    int column;

    VarLocation(std::string file, int line, int column)
        : file(file), line(line), column(column) {}

    VarLocation(const FullSourceLoc &fullLoc) {
        requireTrue(fullLoc.hasManager(), "no source manager!");
        requireTrue(fullLoc.isValid(), "invalid location!");

        file = fullLoc.getFileEntry()->tryGetRealPathName();
        line = fullLoc.getLineNumber();
        column = fullLoc.getColumnNumber();
        requireTrue(!file.empty(), "empty file path!");
    }

    bool operator==(const VarLocation &other) const {
        return file == other.file && line == other.line &&
               column == other.column;
    }
};

/**
 * Visit all DeclRefExprs and print their parents.
 */
class FindVarVisitor : public RecursiveASTVisitor<FindVarVisitor> {
  private:
    ASTContext *Context;
    VarLocation targetLoc;

  public:
    explicit FindVarVisitor(ASTContext *Context, VarLocation targetLoc)
        : Context(Context), targetLoc(targetLoc) {}

    template <typename NodeT> void visitParentsRecursively(const NodeT &base) {
        const DynTypedNodeList &parents = Context->getParents(base);
        requireTrue(parents.size() == 1, "parent size is not 1");

        if (const Stmt *parent = parents.begin()->get<Stmt>()) {
            llvm::errs() << "    " << parent->getStmtClassName() << "\n";
            if (isa<CompoundStmt>(parent))
                return;
            visitParentsRecursively(*parent);
        } else if (const Decl *parent = parents.begin()->get<Decl>()) {
            llvm::errs() << "    " << parent->getDeclKindName() << "\n";
            visitParentsRecursively(*parent);
        } else {
            llvm::errs() << "    unknown parent\n";
            exit(1);
        }
    }

    bool findMatch(const Stmt *S, const NamedDecl *decl,
                   const SourceLocation &loc) {
        FullSourceLoc FullLocation = Context->getFullLoc(loc);
        VarLocation varLoc(FullLocation);
        bool match = varLoc == targetLoc;
        if (match) {
            const auto &var = decl->getQualifiedNameAsString();
            llvm::errs() << "Found VarDecl: " << var << "\n";
            llvm::errs() << "  at " << varLoc.file << ":" << varLoc.line << ":"
                         << varLoc.column << "\n";
            llvm::errs() << "  parents:\n";
            visitParentsRecursively(*S);
        }

        return match;
    }

    bool VisitDeclStmt(DeclStmt *ds) {
        for (Decl *d : ds->decls()) {
            if (VarDecl *vd = dyn_cast<VarDecl>(d)) {
                findMatch(ds, vd, vd->getLocation());
            }
        }
        return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *dre) {
        findMatch(dre, dre->getDecl(), dre->getBeginLoc());
        return true;
    }
};

struct Graph {
    int n; // 0-indexed
    std::vector<std::vector<int>> G;
    std::vector<int> d;

    const int INF = 0x3f3f3f3f;

    struct Node {
        int u, d;
        bool operator<(const Node &b) const { return d > b.d; }
    };

    Graph(int n) : n(n) {
        G.resize(n);
        d.resize(n);
    }

    void addEdge(int u, int v) { G[u].push_back(v); }

    void dij(int s) {
        std::fill(d.begin(), d.end(), INF);
        d[s] = 0;

        std::priority_queue<Node> q;
        q.push({s, 0});

        while (!q.empty()) {
            Node p = q.top();
            q.pop();
            int u = p.u;
            if (p.d != d[u])
                continue;
            for (int v : G[u]) {
                if (d[v] > d[u] + 1) {
                    d[v] = d[u] + 1;
                    q.push({v, d[v]});
                }
            }
        }
    }
};

struct BlockGraph {
    const ASTContext &Context;
    const CFG *cfg;
    Graph g;

    std::map<const Stmt *, int> blockIdOfStmt;

    BlockGraph(const ASTContext &Context, const CFG *cfg)
        : Context(Context), cfg(cfg), g(cfg->size()) {
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;

            llvm::errs() << "Block " << B.getBlockID() << ":\n";

            // map stmts to block ids
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    const Stmt &S = *CS->getStmt();
                    blockIdOfStmt[&S] = B.getBlockID();

                    llvm::errs() << "  " << S.getStmtClassName() << " ("
                                 << S.getID(Context) << ")\n";
                }
            }

            llvm::errs() << "  successors:";
            // add edges
            for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
                const CFGBlock *Succ = *SI;
                g.addEdge(B.getBlockID(), Succ->getBlockID());

                llvm::errs() << " " << Succ->getBlockID();
            }
            llvm::errs() << "\n";
        }
    }
};

struct FunctionInfo {
    const FunctionDecl *D;
    std::string name;
    std::string file;
    int line;
    int column;

    const CFG *cfg;
    BlockGraph *bg;

    static FunctionInfo *fromDecl(FunctionDecl *D) {
        // ensure that the function has a body
        if (!D->hasBody())
            return nullptr;

        // get location
        FullSourceLoc FullLocation =
            D->getASTContext().getFullLoc(D->getBeginLoc());
        if (FullLocation.isInvalid() || !FullLocation.hasManager())
            return nullptr;

        std::string name = D->getQualifiedNameAsString();
        int line = FullLocation.getSpellingLineNumber();
        int column = FullLocation.getSpellingColumnNumber();
        StringRef file = FullLocation.getFileEntry()->tryGetRealPathName();
        requireTrue(!file.empty());

        // build CFG
        const std::unique_ptr<CFG> cfg = CFG::buildCFG(
            D, D->getBody(), &D->getASTContext(), CFG::BuildOptions());

        // build graph for each CFGBlock
        BlockGraph *bg = new BlockGraph(D->getASTContext(), cfg.get());

        FunctionInfo *fi = new FunctionInfo();
        fi->D = D;
        fi->name = name;
        fi->file = file;
        fi->line = line;
        fi->column = column;
        fi->cfg = cfg.get();
        fi->bg = bg;
        return fi;
    }
};

class FindPathVisitor : public RecursiveASTVisitor<FindPathVisitor> {
  private:
    ASTContext *Context;
    std::set<const FunctionDecl *> functionDecls;
    std::map<const FunctionDecl *, const FunctionInfo *> infoOfFunction;

  public:
    explicit FindPathVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitFunctionDecl(FunctionDecl *D) {

        FunctionInfo *fi = FunctionInfo::fromDecl(D);
        if (fi == nullptr)
            return true;

        functionDecls.insert(D);
        infoOfFunction[D] = fi;

        return true;
    }

    void collect(/**/) {
        for (const FunctionDecl *D : functionDecls) {
            llvm::errs() << "------ FunctionDecl: "
                         << D->getQualifiedNameAsString() << "\n";
            const FunctionInfo *fi = infoOfFunction[D];
        }
    }
};

/**
 * Visit all FunctionDecls and print their CFGs.
 */
class FunctionDeclVisitor : public RecursiveASTVisitor<FunctionDeclVisitor>,
                            public HasContextVisitor {
  public:
    explicit FunctionDeclVisitor(ASTContext *Context)
        : HasContextVisitor(Context) {}

    bool VisitFunctionDecl(FunctionDecl *D) {
        FullSourceLoc FullLocation = Context->getFullLoc(D->getBeginLoc());
        requireTrue(FullLocation.hasManager(), "no source manager!");
        if (FullLocation.isInvalid())
            return true;

        llvm::errs() << "------ FunctionDecl: " << D->getQualifiedNameAsString()
                     << " at " << FullLocation.getSpellingLineNumber() << ":"
                     << FullLocation.getSpellingColumnNumber() << "\n";

        if (!D->hasBody())
            return true;

        // show call graph
        // TranslationUnitDecl *TUD = Context->getTranslationUnitDecl();
        // CallGraph CG;
        // CG.addToCallGraph(TUD);
        // CG.viewGraph();

        llvm::errs() << "--------- CFG dump: " << D->getQualifiedNameAsString()
                     << "\n";
        // build CFG
        auto cfg = CFG::buildCFG(D, D->getBody(), &D->getASTContext(),
                                 CFG::BuildOptions());
        cfg->dump(D->getASTContext().getLangOpts(), true);
        cfg->viewCFG(D->getASTContext().getLangOpts());

        int n = cfg->size(); // num of blocks
        llvm::errs() << "Num of blocks: " << n << "\n";

        // traverse each block
        llvm::errs() << "--------- Block traversal: "
                     << D->getQualifiedNameAsString() << "\n";
        for (auto BI = cfg->begin(); BI != cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            // print block ID
            llvm::errs() << "Block " << B.getBlockID();
            if (&B == &cfg->getEntry()) {
                llvm::errs() << " (Entry)";
            } else if (&B == &cfg->getExit()) {
                llvm::errs() << " (Exit)";
            }
            llvm::errs() << ":\n";

            // traverse & print block contents
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                llvm::errs() << "  ";
                E.dump();
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    // CS->getStmt()->dump();
                    // printStmtLocation(*CS->getStmt());
                }
            }

            // print block terminator
            if (B.getTerminator().isValid()) {
                const CFGTerminator &T = B.getTerminator();
                if (T.getStmt()) {
                    const Stmt &S = *T.getStmt();
                    llvm::errs() << "  T: <" << S.getStmtClassName() << ">\n";
                    // printStmtLocation(S);
                }
            }

            // print predecessors
            llvm::errs() << "  Preds:";
            for (auto PI = B.pred_begin(); PI != B.pred_end(); ++PI) {
                const CFGBlock *Pred = *PI;
                llvm::errs() << " B" << Pred->getBlockID();
            }
            llvm::errs() << "\n";

            // print successors
            llvm::errs() << "  Succs:";
            for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
                const CFGBlock *Succ = *SI;
                llvm::errs() << " B" << Succ->getBlockID();
            }
            llvm::errs() << "\n";
        }

        return true;
    }
};

class FindNamedClassConsumer : public clang::ASTConsumer {
  private:
    fs::path currentFile;

  public:
    explicit FindNamedClassConsumer(fs::path currentFile)
        : currentFile(currentFile) {
        requireTrue(currentFile.is_absolute());
        llvm::errs() << "--- Processing " << currentFile << "\n";
    }

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
        auto *TUD = Context.getTranslationUnitDecl();
        llvm::errs() << "\n--- TranslationUnitDecl Dump ---\n";
        TUD->dump();

        // call different visitors
        // llvm::errs() << "\n--- FunctionDeclVisitor ---\n";
        // FunctionDeclVisitor(&Context).TraverseDecl(TUD);

        llvm::errs() << "\n--- FindVarVisitor ---\n";
        VarLocation targetLoc(
            "/home/thebesttv/vul/llvm-project/graph-generation/test4.cpp", 2,
            7);
        FindVarVisitor(&Context, targetLoc).TraverseDecl(TUD);

        targetLoc.line = 2;
        targetLoc.column = 14;
        FindVarVisitor(&Context, targetLoc).TraverseDecl(TUD);

        targetLoc.line = 23;
        targetLoc.column = 7;
        FindVarVisitor(&Context, targetLoc).TraverseDecl(TUD);

        targetLoc.line = 23;
        targetLoc.column = 11;
        FindVarVisitor(&Context, targetLoc).TraverseDecl(TUD);

        targetLoc.line = 23;
        targetLoc.column = 15;
        FindVarVisitor(&Context, targetLoc).TraverseDecl(TUD);

        llvm::errs() << "\n--- FindPathVisitor ---\n";
        FindPathVisitor fpv(&Context);
        fpv.TraverseDecl(TUD);
        fpv.collect();
    }
};

class FindNamedClassAction : public clang::ASTFrontendAction {
  public:
    virtual std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &Compiler,
                      llvm::StringRef InFile) {
        fs::path currentFile = fs::canonical(BUILD_PATH / InFile.str());
        return std::make_unique<FindNamedClassConsumer>(currentFile);
    }
};

std::unique_ptr<CompilationDatabase>
getCompilationDatabase(fs::path buildPath) {
    llvm::errs() << "Getting compilation database from: " << buildPath << "\n";
    std::string errorMsg;
    std::unique_ptr<CompilationDatabase> cb =
        CompilationDatabase::autoDetectFromDirectory(buildPath.string(),
                                                     errorMsg);
    if (!cb) {
        llvm::errs() << "Error while trying to load a compilation database:\n"
                     << errorMsg << "Running without flags.\n";
        exit(1);
    }
    return cb;
}

int main(int argc, const char **argv) {
    BUILD_PATH = fs::canonical(fs::absolute(argv[1]));
    std::unique_ptr<CompilationDatabase> cb =
        getCompilationDatabase(BUILD_PATH);

    const auto &allFiles = cb->getAllFiles();

    llvm::errs() << "All files:\n";
    for (auto &file : allFiles) {
        llvm::errs() << "  " << file << "\n";
    }

    ClangTool Tool(*cb, allFiles);
    return Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());
}