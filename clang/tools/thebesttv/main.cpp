#include "FunctionInfo.h"
#include "GenICFG.h"
#include "ICFG.h"
#include "PathFinder.h"
#include "VarFinder.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"

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

void deduplicateCompilationDatabase(fs::path path) {
    std::ifstream ifs(path);
    ordered_json input = ordered_json::parse(ifs);
    ifs.close();

    bool hasDuplicate = false;
    std::set<std::string> visitedFiles;
    ordered_json output;
    for (auto &cmd : input) {
        std::string file = cmd["file"];
        if (visitedFiles.find(file) != visitedFiles.end()) {
            logger.warn("Duplicate entry for file: {}", file);
            hasDuplicate = true;
            continue;
        }
        visitedFiles.insert(file);
        output.push_back(cmd);
    }

    if (!hasDuplicate) {
        logger.info("No duplicate entries found in compilation database");
        return;
    }

    logger.warn("Found duplicate entries in compilation database!");
    {
        fs::path backup = path.string() + ".bk";
        logger.warn("Original database is backed up to: {}", backup);
        std::ofstream o(backup);
        o << input.dump(4, ' ', false, json::error_handler_t::replace)
          << std::endl;
        o.close();
    }
    {
        std::ofstream o(path);
        o << output.dump(4, ' ', false, json::error_handler_t::replace)
          << std::endl;
        o.close();
        logger.warn("Deduplicated database is in: {}", path);
    }
}

std::unique_ptr<CompilationDatabase>
getCompilationDatabase(fs::path buildPath) {
    logger.info("Reading compilation database from: {}", buildPath);
    std::string errorMsg;
    std::unique_ptr<CompilationDatabase> cb =
        CompilationDatabase::autoDetectFromDirectory(buildPath.string(),
                                                     errorMsg);
    // FIXME: 在 npe/input.json 下会报错
    // #include "clang/Tooling/JSONCompilationDatabase.h"
    // JSONCompilationDatabase::loadFromFile(
    //     buildPath.string(), errorMsg, JSONCommandLineSyntax::AutoDetect);
    if (!cb) {
        logger.error("Error while trying to load compilation database: {}",
                     errorMsg);
        exit(1);
    }
    logger.info("There are {} entries for {} files in total",
                cb->getAllCompileCommands().size(), cb->getAllFiles().size());
    return cb;
}

struct VarLocResult {
    int fid, bid;

    VarLocResult() : fid(-1), bid(-1) {}
    VarLocResult(int fid, int bid) : fid(fid), bid(bid) {}
    VarLocResult(const FunctionInfo *fi, const CFGBlock *block)
        : fid(Global.getIdOfFunction(fi->signature)), bid(block->getBlockID()) {
    }

    bool isValid() const { return fid != -1; }
};

/**
 * requireExact: 只对路径有效，表示是否需要精确匹配
 */
VarLocResult locateVariable(const fif &functionsInFile, const std::string &file,
                            int line, int column, bool isStmt,
                            bool requireExact = true) {
    FindVarVisitor visitor;

    for (const FunctionInfo *fi : functionsInFile.at(file)) {
        // function is defined later than targetLoc
        if (fi->line > line)
            continue;

        // search all CFG stmts in function for matching variable
        ASTContext *Context = &fi->D->getASTContext();
        for (const auto &[stmt, block] : fi->stmtBlockPairs) {
            if (isStmt) {
                // search for stmt
                auto bLoc =
                    Location::fromSourceLocation(*Context, stmt->getBeginLoc());
                auto eLoc =
                    Location::fromSourceLocation(*Context, stmt->getEndLoc());

                if (!bLoc || bLoc->file != file)
                    continue;

                // 精确匹配：要求行号、列号相同
                bool matchExact = bLoc->line == line && bLoc->column == column;
                // 模糊匹配：行号在语句 begin 和 end 之间即可
                bool matchInexact = bLoc->line <= line && line <= eLoc->line;

                if ((requireExact && matchExact) ||
                    (!requireExact && matchInexact)) {
                    int id = block->getBlockID();
                    logger.info("Found stmt in {} B{} at {}:{}:{}",
                                fi->signature, id, file, line, column);
                    return VarLocResult(fi, block);
                }
            } else {
                // search for var within stmt
                const std::string var =
                    visitor.findVarInStmt(Context, stmt, file, line, column);
                if (!var.empty()) {
                    int id = block->getBlockID();
                    logger.info("Found var '{}' in {} block {} at {}:{}:{}",
                                var, fi->signature, id, file, line, column);
                    return VarLocResult(fi, block);
                }
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

VarLocResult locateVariable(const FunctionLocator &locator, const Location &loc,
                            bool isStmt) {
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

    auto exactResult = locateVariable(functionsInFile, loc.file, loc.line,
                                      loc.column, isStmt, true);
    if (exactResult.isValid())
        return exactResult;
    // 精确匹配失败
    logger.warn("Unable to find exact match! Trying inexact matching...");
    logger.warn("  {}:{}:{}", loc.file, loc.line, loc.column);
    return locateVariable(functionsInFile, loc.file, loc.line, loc.column,
                          isStmt, false);
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

    SourceLocation b = range.getBegin();
    if (b.isMacroID()) {
        b = SM.getExpansionLoc(b);
    }
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
    SourceLocation _e = range.getEnd();
    if (_e.isMacroID()) {
        // _e = SM.getExpansionLoc(_e);
        _e = getEndOfMacroExpansion(_e, Context);
    }
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

    logger.info(">> Node {} is in {} B{}", u, loc.name, bid);

    ClangTool Tool(*Global.cb, {loc.file});
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    std::vector<std::unique_ptr<ASTUnit>> ASTs;
    Tool.buildASTs(ASTs);

    // FIXME: compile_commands
    // 中一个文件可能对应多条编译命令，所以这里可能有多个AST
    // TODO: 想办法记录 function 用的是哪一条命令，然后只用这一条生成的
    requireTrue(ASTs.size() != 0, "No AST for file");
    if (ASTs.size() > 1) {
        logger.warn("Warning: multiple ASTs for file {}", loc.file);
    }
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

/**
 * 删除路径中连续重复的 stmt
 */
void deduplicateAndFixLocations(ordered_json &locations, int fromLine,
                                int toLine) {
    std::set<std::string> interestedFields = {
        "type", "file", "beginLine", "beginColumn", "endLine", "endColumn"};

    std::deque<ordered_json> result;
    ordered_json lastEntry;
    for (const auto &j : locations) {
        if (j["type"] != "stmt") {
            result.push_back(j);
            lastEntry.clear();
            continue;
        }

        if (allFieldsMatch(j, lastEntry, interestedFields)) {
            logger.warn("Skipping duplicated stmt in path (most likely due to "
                        "macro expansion)");
            continue;
        }

        result.push_back(j);
        lastEntry = j;
    }

    // 输出会包含 BB 中的所有语句。
    // 如果 source 不是 BB 中第一条，它前面的语句也会被输出。
    // 这里删除路径中，可能存在的 source 之前的、sink 之后的语句。
    while (!result.empty() && result.front()["beginLine"] < fromLine)
        result.pop_front();
    while (!result.empty() && result.back()["beginLine"] > toLine)
        result.pop_back();

    locations = result;
}

void saveAsJson(int fromLine, int toLine,
                const std::set<std::vector<int>> &results,
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
        ordered_json jPath, locations;
        jPath["type"] = type;
        jPath["nodes"] = path;
        for (int x : path) {
            dumpICFGNode(x, locations);
        }
        deduplicateAndFixLocations(locations, fromLine, toLine);
        jPath["locations"] = locations;
        jResults.push_back(jPath);
    }
}

static void findPathBetween(const VarLocResult &from, int fromLine,
                            VarLocResult to, int toLine,
                            const std::vector<VarLocResult> &_pointsToPass,
                            const std::vector<VarLocResult> &_pointsToAvoid,
                            const std::string &type, ordered_json &jResults) {
    requireTrue(from.isValid(), "FROM location is invalid");
    requireTrue(to.isValid(), "TO location is invalid");

    ICFG &icfg = Global.icfg;
    int u = icfg.getNodeId(from.fid, from.bid);
    int v = icfg.getNodeId(to.fid, to.bid);

    std::vector<int> pointsToPass;
    for (const auto &loc : _pointsToPass) {
        requireTrue(loc.isValid());
        pointsToPass.push_back(icfg.getNodeId(loc.fid, loc.bid));
    }
    std::set<int> pointsToAvoid;
    for (const auto &loc : _pointsToAvoid) {
        requireTrue(loc.isValid());
        pointsToAvoid.insert(icfg.getNodeId(loc.fid, loc.bid));
    }

    auto pFinder = DijPathFinder(icfg);
    pFinder.search(u, v, pointsToPass, pointsToAvoid, 3);

    saveAsJson(fromLine, toLine, pFinder.results, type, jResults);
}

void handleInputEntry(const VarLocResult &from, int fromLine, VarLocResult to,
                      int toLine, const std::vector<VarLocResult> &path,
                      const std::string &type, ordered_json &jResults) {
    int fromFid = from.fid;

    VarLocResult sourceExit(fromFid,
                            Global.icfg.entryExitOfFunction[fromFid].second);
    // 位于 source 所在函数的路径
    std::vector<VarLocResult> pathInSourceFunction;
    for (auto &p : path)
        if (p.fid == fromFid)
            pathInSourceFunction.push_back(p);

    if (type == "npe") {
        logger.info("Handle known type: {}", type);

        logger.info("Generating NPE bug version ...");
        findPathBetween(from, fromLine, to, toLine, path, {}, "npe-bug",
                        jResults);

        logger.info("Generating NPE fix version ...");
        findPathBetween(
            from, fromLine,
            // 如果中间路径中没有 source 所在函数中的语句，就用 exit
            // 否则，用中间路径的最后一条
            pathInSourceFunction.empty() ? sourceExit
                                         : pathInSourceFunction.back(),
            INT_MAX, pathInSourceFunction, {to}, "npe-fix", jResults);
    } else {
        logger.info("Handle unknown type: {}", type);
        if (!to.isValid()) {
            logger.warn("Missing sink! Using exit of source instead");
            to = sourceExit;
        }
        findPathBetween(from, fromLine, to, toLine, path, {}, type, jResults);
    }
}

/**
 * 生成全程序调用图
 */
void generateICFG(const std::vector<std::string> &allFiles) {
    logger.info("--- Generating whole program call graph ---");
    ClangTool Tool(*Global.cb, allFiles);
    DiagnosticConsumer DC = IgnoringDiagConsumer();
    Tool.setDiagnosticConsumer(&DC);

    Tool.run(newFrontendActionFactory<GenICFGAction>().get());
}

void generatePathFromOneEntry(const ordered_json &result,
                              FunctionLocator &locator, ordered_json &output) {
    std::string type = result["type"].template get<std::string>();

    const ordered_json &locations = result["locations"];
    VarLocResult from, to;
    int fromLine, toLine;
    std::vector<VarLocResult> path;
    for (const ordered_json &loc : locations) {
        std::string type = loc["type"].template get<std::string>();
        if (type != "source" && type != "sink" && type != "stmt") {
            logger.warn("Skipping path type: {}", type);
            continue;
        }

        // 目前把 source 和 sink 都当作 stmt 来处理，
        // 精确匹配不上的话，就模糊匹配
        bool isStmt = true; // type == "stmt";
        Location jsonLoc(loc);
        VarLocResult varLoc = locateVariable(locator, jsonLoc, isStmt);
        if (!varLoc.isValid()) {
            logger.error(
                "Error: cannot locate {} at {}", type,
                loc.dump(4, ' ', false, json::error_handler_t::replace));
            throw std::runtime_error("Can't locate input entry");
        }

        if (type == "source") {
            from = varLoc;
            fromLine = jsonLoc.line;
        } else if (type == "sink") {
            to = varLoc;
            toLine = jsonLoc.line;
            // sink is the last stmt
            break;
        } else {
            // source is the first stmt
            if (!from.isValid())
                continue;
            path.emplace_back(varLoc);
        }
    }

    handleInputEntry(from, fromLine, to, toLine, path, type, output["results"]);
}

void generateFromInput(const ordered_json &input, fs::path outputDir) {
    logger.info("--- Path-finding ---");

    FunctionLocator locator;
    fs::path jsonResult = outputDir / "output.json";
    logger.info("Result will be saved to: {}", jsonResult);

    int total = input["results"].size();
    logger.info("There are {} results to search", total);

    ordered_json output(input);
    output["results"].clear();

    int cnt = 0;
    for (const ordered_json &result : input["results"]) {
        cnt++;
        std::string type = result["type"].template get<std::string>();
        logger.info("[{}/{}] type: {}", cnt, total, type);
        try {
            generatePathFromOneEntry(result, locator, output);
        } catch (const std::exception &e) {
            logger.error("Exception encountered: {}", e.what());
        }
    }

    std::ofstream o(jsonResult);
    o << output.dump(4, ' ', false, json::error_handler_t::replace)
      << std::endl;
    o.close();
}

int main(int argc, const char **argv) {
    spdlog::set_level(spdlog::level::debug);

    if (argc != 2) {
        logger.error("Usage: {} IR.json", argv[0]);
        return 1;
    }

    llvm::InitLLVM X(argc, argv);

    fs::path jsonPath = fs::absolute(argv[1]);
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        logger.error("Cannot open file {}", jsonPath);
        return 1;
    }
    logger.info("Reading from input json: {}", jsonPath);
    ordered_json input = ordered_json::parse(ifs);

    Global.projectDirectory =
        fs::canonical(input["root"].template get<std::string>()).string();

    fs::path compile_commands =
        fs::canonical(input["compile_commands"].template get<std::string>());
    logger.info("Compilation database: {}", compile_commands);
    deduplicateCompilationDatabase(compile_commands);
    Global.cb = getCompilationDatabase(compile_commands);

    generateICFG(Global.cb->getAllFiles());

    {
        int m = 0;
        for (const auto &edges : Global.icfg.G) {
            m += edges.size();
        }
        logger.info("ICFG: {} nodes, {} edges", Global.icfg.n, m);
    }

    fs::path outputDir = jsonPath.parent_path();

    generateFromInput(input, outputDir);

    return 0;

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
}