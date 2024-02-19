#include "FunctionInfo.h"
#include "GenICFG.h"
#include "ICFG.h"
#include "PathFinder.h"
#include "VarFinder.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "llvm/Support/Program.h"

class FunctionAccumulator : public RecursiveASTVisitor<FunctionAccumulator> {
  private:
    fif &functionsInFile;

  public:
    explicit FunctionAccumulator(fif &functionsInFile)
        : functionsInFile(functionsInFile) {}

    bool VisitFunctionDecl(FunctionDecl *D) {
        FunctionInfo *fi = FunctionInfo::fromDecl(D);
        if (fi == nullptr)
            return true;

        functionsInFile[fi->file].insert(fi);

        return true;
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

struct VarLocResult {
    const int fid, bid;

    VarLocResult() : fid(-1), bid(-1) {}
    VarLocResult(const FunctionInfo *fi, const CFGBlock *block)
        : fid(Global.getIdOfFunction(fi->signature)), bid(block->getBlockID()) {
    }

    bool isValid() const { return fid != -1; }
};

VarLocResult locateVariable(const fif &functionsInFile, const std::string &file,
                            int line, int column) {
    FindVarVisitor visitor;

    for (const FunctionInfo *fi : functionsInFile.at(file)) {
        // function is defined later than targetLoc
        if (fi->line > line)
            continue;

        // search all CFG stmts in function for matching variable
        ASTContext *Context = &fi->D->getASTContext();
        for (const auto &[stmt, block] : fi->stmtBlockPairs) {
            const std::string var =
                visitor.findVarInStmt(Context, stmt, file, line, column);
            if (!var.empty()) {
                int id = block->getBlockID();
                llvm::errs() << "Found var '" << var << "' in " << fi->signature
                             << " at " << line << ":" << column << " in block "
                             << id << "\n";
                return VarLocResult(fi, block);
            }
        }
    }
    return VarLocResult();
}

struct FunctionLocator {
    // file -> (fid, start line)
    std::map<std::string, std::vector<std::pair<int, int>>> functionLocations;

    FunctionLocator() {
        for (int i = 0; i < Global.functionLocations.size(); i++) {
            const Location &loc = Global.functionLocations[i];
            functionLocations[loc.file].emplace_back(i, loc.line);
        }
        // 根据 line 降序排列
        for (auto &[file, locs] : functionLocations) {
            std::sort(locs.begin(), locs.end(),
                      [](const auto &a, const auto &b) {
                          return a.second > b.second;
                      });
        }
    }

    int getFid(const Location &loc) const {
        auto it = functionLocations.find(loc.file);
        if (it == functionLocations.end())
            return -1;

        for (const auto &[fid, startLine] : it->second) {
            if (loc.line >= startLine) {
                return fid;
            }
        }
        return -1;
    }
};

VarLocResult locateVariable(const FunctionLocator &locator,
                            const Location &loc) {
    int fid = locator.getFid(loc);
    if (fid == -1) {
        return VarLocResult();
    }

    ClangTool Tool(*Global.cb, {loc.file});
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    fif functionsInFile;
    for (auto &AST : ASTs) {
        ASTContext &Context = AST->getASTContext();
        auto *TUD = Context.getTranslationUnitDecl();
        if (TUD->isUnavailable())
            continue;
        FunctionAccumulator(functionsInFile).TraverseDecl(TUD);
    }

    return locateVariable(functionsInFile, loc.file, loc.line, loc.column);
}

std::string getSourceCode(SourceManager &SM, const SourceRange &range) {
    const auto &_b = range.getBegin();
    const auto &_e = range.getEnd();
    const char *b = SM.getCharacterData(_b);
    const char *e = SM.getCharacterData(_e);

    return std::string(b, e - b);
}

void saveLocationInfo(ASTContext &Context, const SourceRange &range,
                      ordered_json &j) {
    SourceManager &SM = Context.getSourceManager();

    const SourceLocation &b = range.getBegin();
    auto bLoc = Location::fromSourceLocation(Context, b);
    if (bLoc) {
        j["file"] = bLoc->file;
        j["beginLine"] = bLoc->line;
        j["beginColumn"] = bLoc->column;
    } else {
        j["file"] = "!!! begin loc invalid !!!";
        j["beginLine"] = -1;
        j["beginColumn"] = -1;
    }

    /**
     * SourceRange中，endLoc是最后一个token的起始位置，所以需要找到这个token的结束位置
     * See:
     * https://stackoverflow.com/a/11154162/11938767
     * https://discourse.llvm.org/t/problem-with-retrieving-the-binaryoperator-rhs-end-location/51897
     * https://clang.llvm.org/docs/InternalsManual.html#sourcerange-and-charsourcerange
     */
    const SourceLocation &_e = range.getEnd();
    SourceLocation e =
        Lexer::getLocForEndOfToken(_e, 0, SM, Context.getLangOpts());
    if (e.isInvalid())
        e = _e;
    auto eLoc = Location::fromSourceLocation(Context, e);

    if (eLoc) {
        j["endLine"] = eLoc->line;
        j["endColumn"] = eLoc->column;
    } else {
        j["endLine"] = -1;
        j["endColumn"] = -1;
    }

    std::string content;
    if (b.isValid() && e.isValid()) {
        const char *cb = SM.getCharacterData(b);
        const char *ce = SM.getCharacterData(e);
        auto length = ce - cb;
        if (length < 0) {
            content = "!!! length < 0 !!!";
        } else {
            if (length > 80) {
                content = std::string(cb, 80);
                content += " ...";
            } else {
                content = std::string(cb, length);
            }
        }
    }
    j["content"] = content;
}

void dumpICFGNode(int u, ordered_json &jPath) {
    auto [fid, bid] = Global.icfg.functionBlockOfNodeId[u];
    requireTrue(fid != -1);

    const NamedLocation &loc = Global.functionLocations[fid];

    llvm::errs() << ">> Node " << u << " is in " << loc.name << " at block "
                 << bid << "\n";

    ClangTool Tool(*Global.cb, {loc.file});
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    requireTrue(ASTs.size() == 1);
    std::unique_ptr<ASTUnit> AST = std::move(ASTs[0]);

    fif functionsInFile;
    ASTContext &Context = AST->getASTContext();
    auto *TUD = Context.getTranslationUnitDecl();
    requireTrue(!TUD->isUnavailable());
    FunctionAccumulator(functionsInFile).TraverseDecl(TUD);

    for (const FunctionInfo *fi : functionsInFile.at(loc.file)) {
        if (fi->signature != Global.functionLocations[fid].name)
            continue;
        for (auto BI = fi->cfg->begin(); BI != fi->cfg->end(); ++BI) {
            const CFGBlock &B = **BI;
            if (B.getBlockID() != bid)
                continue;

            bool isEntry = (&B == &fi->cfg->getEntry());
            bool isExit = (&B == &fi->cfg->getExit());
            if (isEntry || isExit) {
                ordered_json j;
                j["type"] = isEntry ? "entry" : "exit";
                // TODO: content只要declaration就行，不然太大了
                saveLocationInfo(Context, fi->D->getSourceRange(), j);
                jPath.push_back(j);
                return;
            }

            // B.dump(fi->cfg, Context.getLangOpts(), true);

            std::vector<const Stmt *> allStmts;
            std::set<const Stmt *> isChild;

            // iterate over all elements to find stmts & record children
            for (auto EI = B.begin(); EI != B.end(); ++EI) {
                const CFGElement &E = *EI;
                if (std::optional<CFGStmt> CS = E.getAs<CFGStmt>()) {
                    const Stmt *S = CS->getStmt();
                    allStmts.push_back(S);

                    // iterate over childern
                    for (const Stmt *child : S->children()) {
                        if (child != nullptr)
                            isChild.insert(child);
                    }
                }
            }

            // print all non-child stmts
            for (const Stmt *S : allStmts) {
                if (isChild.find(S) != isChild.end())
                    continue;
                // S is not child of any stmt in this CFGBlock

                // auto bLoc =
                //     Location::fromSourceLocation(Context, S->getBeginLoc());
                // auto eLoc =
                //     Location::fromSourceLocation(Context, S->getEndLoc());
                // llvm::errs()
                //     << "  Stmt " << bLoc->line << ":" << bLoc->column << " "
                //     << eLoc->line << ":" << eLoc->column << "\n";
                // S->dumpColor();

                ordered_json j;
                j["type"] = "stmt";
                saveLocationInfo(Context, S->getSourceRange(), j);
                j["stmtKind"] = std::string(S->getStmtClassName());
                jPath.push_back(j);
            }

            return;
        }
    }
}

void saveAsJson(const std::set<std::vector<int>> &results,
                const std::string &type, ordered_json &jResults) {
    std::vector<std::vector<int>> sortedResults(results.begin(), results.end());
    // sort based on length
    std::sort(sortedResults.begin(), sortedResults.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) {
                  return a.size() < b.size();
              });
    int cnt = 0;
    for (const auto &path : sortedResults) {
        if (cnt++ > 10)
            break;
        ordered_json jPath;
        jPath["type"] = type;
        for (int x : path) {
            dumpICFGNode(x, jPath["locations"]);
        }
        jResults.push_back(jPath);
    }
}

static void findPathBetween(const VarLocResult &from, const VarLocResult &to,
                            const std::string &type, ordered_json &jResults) {
    if (!from.isValid() || !to.isValid()) {
        llvm::errs() << "Invalid variable location!\n";
        return;
    }

    ICFG &icfg = Global.icfg;
    int u = icfg.getNodeId(from.fid, from.bid);
    int v = icfg.getNodeId(to.fid, to.bid);

    llvm::errs() << "u: " << u << ", v: " << v << "\n";

    ICFGPathFinder pFinder(icfg);
    pFinder.search(u, v, 3);

    saveAsJson(pFinder.results, type, jResults);
}

/**
 * 生成全程序调用图
 */
void generateICFG(const std::vector<std::string> &allFiles) {
    llvm::errs() << "\n--- Generating whole program call graph ---\n";
    ClangTool Tool(*Global.cb, allFiles);
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    Tool.run(newFrontendActionFactory<GenICFGAction>().get());
}

/**
 * FIXME: all files 应该只有
 * .c/.cpp，不包括头文件，所以统计会不准。以及，buildPath
 * 现在是一个 JSON 文件了，不是目录。
 */
void printCloc(const std::vector<std::string> &allFiles) {
    // save all files to "compile_files.txt" under build path
    fs::path resultFiles =
        fs::path(Global.projectDirectory) / "compile_files.txt";
    std::ofstream ofs(resultFiles);
    if (!ofs.is_open()) {
        llvm::errs() << "Error: cannot open file " << resultFiles << "\n";
        exit(1);
    }
    for (auto &file : allFiles)
        ofs << file << "\n";
    ofs.close();

    // run cloc on all files
    if (ErrorOr<std::string> P = sys::findProgramByName("cloc")) {
        std::string programPath = *P;
        std::vector<StringRef> args;
        args.push_back("cloc");
        args.push_back("--list-file");
        args.push_back(resultFiles.c_str()); // don't use .string() here
        std::string errorMsg;
        if (sys::ExecuteAndWait(programPath, args, std::nullopt, {}, 0, 0,
                                &errorMsg)) {
            llvm::errs() << "Error: " << errorMsg << "\n";
        }
    }
}

int main(int argc, const char **argv) {
    if (argc != 2) {
        llvm::errs() << "Usage: " << argv[0] << " IR.json\n";
        return 1;
    }

    fs::path jsonPath = fs::absolute(argv[1]);
    llvm::errs() << "Reading from json: " << jsonPath << "\n";
    std::ifstream ifs(jsonPath);
    ordered_json input = ordered_json::parse(ifs);

    Global.projectDirectory =
        fs::canonical(input["root"].template get<std::string>()).string();

    fs::path compile_commands =
        fs::canonical(input["compile_commands"].template get<std::string>());
    llvm::errs() << "compile_commands: " << compile_commands << "\n";
    Global.cb = getCompilationDatabase(compile_commands);

    // print all files in compilation database
    const auto &allFiles = Global.cb->getAllFiles();
    llvm::errs() << "All files (" << allFiles.size() << "):\n";
    for (auto &file : allFiles)
        llvm::errs() << "  " << file << "\n";

    generateICFG(allFiles);

    {
        llvm::errs() << "--- ICFG ---\n";
        llvm::errs() << "  n: " << Global.icfg.n << "\n";
        int m = 0;
        for (const auto &edges : Global.icfg.G) {
            m += edges.size();
        }
        llvm::errs() << "  m: " << m << "\n";
    }

    fs::path outputDir = jsonPath.parent_path();
    {
        llvm::errs() << "--- Path-finding ---\n";

        FunctionLocator locator;
        fs::path jsonResult = outputDir / "output.json";
        llvm::errs() << "Output: " << jsonResult << "\n";

        int total = input["results"].size();
        llvm::errs() << "There are " << total << " results to search.\n";

        ordered_json output(input);
        output["results"].clear();

        int cnt = 0;
        for (ordered_json &result : input["results"]) {
            cnt++;
            std::string type = result["type"].template get<std::string>();
            llvm::errs() << "[" << cnt << "/" << total << "] " << type << "\n";

            ordered_json &locations = result["locations"];
            Location from, to;
            for (const ordered_json &loc : locations) {
                std::string type = loc["type"].template get<std::string>();
                if (type == "source") {
                    from = Location(loc);
                } else if (type == "sink") {
                    to = Location(loc);
                }
            }

            findPathBetween(locateVariable(locator, from),
                            locateVariable(locator, to), type,
                            output["results"]);
        }

        std::ofstream o(jsonResult);
        o << output.dump(4, ' ', false, json::error_handler_t::replace)
          << std::endl;
        o.close();
    }

    // std::string source = "IOPriorityPanel_new(IOPriority)";
    // std::string target = "Panel_setSelected(Panel *, int)";
    // findPathBetween(locateVariable(source, 23, 11),
    //                 locateVariable(target, 207, 10));

    while (true) {
        std::string methodName;
        llvm::errs() << "> ";
        std::getline(std::cin, methodName);
        if (!std::cin)
            break;
        if (methodName.find('(') == std::string::npos)
            methodName += "(";

        for (const auto &[caller, callees] : Global.callGraph) {
            if (caller.find(methodName) != 0)
                continue;
            llvm::errs() << caller << "\n";
            for (const auto &callee : callees) {
                llvm::errs() << "  " << callee << "\n";
            }
        }
    }

    return 0;
}