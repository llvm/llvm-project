#include "CompilationDatabase.h"
#include "DumpPath.h"
#include "FunctionInfo.h"
#include "GenAST.h"
#include "GenICFG.h"
#include "ICFG.h"
#include "PathFinder.h"
#include "VarFinder.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "lib/args.hxx"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"

class FunctionAccumulator : public RecursiveASTVisitor<FunctionAccumulator> {
  private:
    fif &functionsInFile;

  public:
    explicit FunctionAccumulator(fif &functionsInFile)
        : functionsInFile(functionsInFile) {}

    bool VisitFunctionDecl(FunctionDecl *D) {
        auto fi = FunctionInfo::fromDecl(D);
        if (fi == nullptr)
            return true;

        functionsInFile[fi->file].insert(std::move(fi));

        return true;
    }
};

void fixCompilationDatabase(fs::path path) {
    std::ifstream ifs(path);
    ordered_json input = ordered_json::parse(ifs);
    ifs.close();

    bool needsUpdate = false;
    std::set<std::string> visitedFiles;
    ordered_json output;
    for (auto &cmd : input) {
        std::string file = cmd["file"];
        if (visitedFiles.find(file) != visitedFiles.end()) {
            logger.warn("Duplicate entry for file: {}", file);
            needsUpdate = true;
            continue;
        }

        std::string dir = cmd["directory"];
        if (!dirExists(dir)) {
            logger.warn("Directory does not exist: {}", dir);
            needsUpdate = true;
            continue;
        }

        visitedFiles.insert(file);
        output.push_back(cmd);
    }

    if (!needsUpdate) {
        logger.info("No entry need fixing in compilation database");
        return;
    }

    logger.warn("Some entries need fixing in compilation database!");
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
        logger.warn("Fixsed database is in: {}", path);
    }
}

struct VarLocResult {
    int fid, bid;

    VarLocResult() : fid(-1), bid(-1) {}
    VarLocResult(int fid, int bid) : fid(fid), bid(bid) {}
    VarLocResult(const std::unique_ptr<FunctionInfo> &fi, const CFGBlock *block)
        : fid(Global.getIdOfFunction(fi->signature, fi->file)),
          bid(block->getBlockID()) {}

    bool isValid() const { return fid != -1; }
};

/**
 * requireExact: 只对路径有效，表示是否需要精确匹配
 */
VarLocResult locateVariable(const fif &functionsInFile, const std::string &file,
                            int line, int column, bool isStmt,
                            bool requireExact = true) {
    FindVarVisitor visitor;

    for (const auto &fi : functionsInFile.at(file)) {
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
                bool matchInexact =
                    eLoc != nullptr && bLoc->line <= line && line <= eLoc->line;

                if ((requireExact && matchExact) ||
                    (!requireExact && matchInexact)) {
                    int id = block->getBlockID();
                    auto result = VarLocResult(fi, block);
                    int nodeId =
                        Global.icfg
                            .nodeIdOfFunctionBlock[{result.fid, result.bid}];
                    logger.info("Found stmt in {} B{} ({}) at {}:{}:{}",
                                fi->signature, id, nodeId, file, line, column);
                    return result;
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

    auto ASTFromFile = getASTOfFile(loc.file);
    auto AST = ASTFromFile->getAST();
    if (!AST)
        return VarLocResult();
    ASTContext &Context = AST->getASTContext();
    auto *TUD = Context.getTranslationUnitDecl();

    fif functionsInFile;
    if (!TUD->isUnavailable())
        FunctionAccumulator(functionsInFile).TraverseDecl(TUD);

    auto exactResult = locateVariable(functionsInFile, loc.file, loc.line,
                                      loc.column, isStmt, true);
    if (exactResult.isValid())
        return exactResult;
    // 精确匹配失败
    logger.warn("Unable to find exact match! Trying inexact matching...");
    logger.warn("  {}:{}:{}", loc.file, loc.line, loc.column);
    auto result = locateVariable(functionsInFile, loc.file, loc.line,
                                 loc.column, isStmt, false);
    return result;
}

void dumpICFGNode(int u, ordered_json &jPath) {
    auto [fid, bid] = Global.icfg.functionBlockOfNodeId[u];
    requireTrue(fid != -1);

    const NamedLocation &loc = Global.functionLocations[fid];

    logger.info(">> Node {} is in {} B{}", u, loc.name, bid);

    // FIXME: compile_commands
    // 中一个文件可能对应多条编译命令，所以这里可能有多个AST
    // TODO: 想办法记录 function 用的是哪一条命令，然后只用这一条生成的

    auto ASTFromFile = getASTOfFile(loc.file);
    auto AST = ASTFromFile->getAST();
    requireTrue(AST != nullptr);

    fif functionsInFile;
    ASTContext &Context = AST->getASTContext();
    auto *TUD = Context.getTranslationUnitDecl();
    requireTrue(!TUD->isUnavailable());
    FunctionAccumulator(functionsInFile).TraverseDecl(TUD);

    for (const auto &fi : functionsInFile.at(loc.file)) {
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

                goto dumpICFGNodeExit;
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

            goto dumpICFGNodeExit;
        }
    }

dumpICFGNodeExit:
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

/*
 * 返回生成路径的个数
 */
int findPathBetween(const VarLocResult &from, int fromLine, VarLocResult to,
                    int toLine, const std::vector<VarLocResult> &_pointsToPass,
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
    return pFinder.results.size();
}

void handleInputEntry(const VarLocResult &from, int fromLine, VarLocResult to,
                      int toLine, const std::vector<VarLocResult> &path,
                      const std::string &type, ordered_json &jResults) {

    auto removeNpeBadSource = [&] {
        // 根据有缺陷的 source，删除可疑的 source
        auto &npeSuspectedSources = Global.npeSuspectedSources;
        for (auto it = npeSuspectedSources.begin();
             it != npeSuspectedSources.end();) {
            const auto &loc = *it;
            const std::string &file = loc["file"];
            int beginLine = loc["beginLine"];
            int endLine = loc["endLine"];

            auto &fromFile = Global.functionLocations[from.fid].file;
            if (beginLine <= fromLine && fromLine <= endLine &&
                file == fromFile) {
                logger.info("Removing suspected good source: {}:{}:{}", file,
                            beginLine, loc["beginColumn"]);
                it = npeSuspectedSources.erase(it);
            } else {
                it++;
            }
        };
    };

    // 获取 loc 所在函数的出口
    auto getExit = [](const VarLocResult &loc) {
        requireTrue(loc.isValid());
        int fid = loc.fid;
        return VarLocResult(fid, Global.icfg.entryExitOfFunction[fid].second);
    };

    if (type == "npe") {
        logger.info("Handle known type: {}", type);
        requireTrue(from.isValid());
        requireTrue(to.isValid());

        logger.info("Generating NPE bug version ...");
        int size = findPathBetween(from, fromLine, to, toLine, path, {},
                                   "npe-bug", jResults);
        if (size == 0)
            logger.warn("Unable to find any path for NPE bug version!");

        // 无缺陷版本：source -> sink 所在函数的出口
        // 尽量符合原始缺陷路径。如果找不到，就一步步减小路径
        logger.info("Generating NPE fix version ...");
        auto sinkExit = getExit(to);
        std::vector<VarLocResult> p = path;
        bool found = false;
        while (true) {
            int result = findPathBetween(from, fromLine, sinkExit, INT_MAX, p,
                                         {to}, "npe-fix", jResults);
            if (result) {
                found = true;
                break;
            }
            if (p.empty()) // no more path to find
                break;
            p.pop_back();
        }
        if (!found)
            logger.warn("Unable to find any path for NPE fix version!");

        removeNpeBadSource();
    } else if (type == "npe-bad-source") {
        logger.info("Removing bad NPE source ...");
        requireTrue(from.isValid());
        removeNpeBadSource();
    } else {
        logger.info("Handle unknown type: {}", type);
        requireTrue(from.isValid());
        if (!to.isValid()) {
            logger.warn("Missing sink! Using exit of source instead");
            to = getExit(from);
            toLine = INT_MAX;
        }
        findPathBetween(from, fromLine, to, toLine, path, {}, type, jResults);
    }
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

        // 跳过项目以外的库函数路径
        if (!Global.isUnderProject(jsonLoc.file)) {
            logger.warn("Skipping lib function in {}", jsonLoc.file);
            continue;
        }

        VarLocResult varLoc = locateVariable(locator, jsonLoc, isStmt);
        if (!varLoc.isValid()) {
            logger.error("Error: cannot locate {} at {}", type, loc);
            // 跳过无法定位的中间路径
            if (type == "stmt")
                continue;
            else
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

    for (const auto &loc : Global.npeSuspectedSources) {
        ordered_json j;
        j["type"] = "npe-good-source";
        j["locations"].push_back(loc);
        output["results"].push_back(j);
    }

    std::ofstream o(jsonResult);
    o << output.dump(4, ' ', false, json::error_handler_t::replace)
      << std::endl;
    o.close();
}

int main(int argc, const char **argv) {
    spdlog::set_level(spdlog::level::debug);

    args::ArgumentParser argParser("Path generation tool\n"
                                   "Example:\n"
                                   "  ./tool npe/input.json\n");

    args::HelpFlag help(argParser, "help", "Display help menu", {'h', "help"});

    args::ValueFlag<int> argPoolSize(
        argParser, "N",
        "AST Pool size (max number of ASTs in memory), default 10", {'p'});

    args::Positional<std::string> argIR(argParser, "IR", "Path to input.json",
                                        {args::Options::Required});

    try {
        argParser.ParseCLI(argc, argv);
    } catch (const args::Help &) {
        std::cout << argParser;
        return 0;
    } catch (const args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << argParser;
        return 1;
    } catch (const args::ValidationError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << argParser;
        return 1;
    }

    logger.info("AST & ICFG generation method: sequential");

    {
        int ASTPoolSize = 10;
        if (argPoolSize) {
            ASTPoolSize = args::get(argPoolSize);
            requireTrue(ASTPoolSize >= 1,
                        "AST pool size must be greater than 0");
        }
        logger.info("AST pool size: {}", ASTPoolSize);
        Global.ASTPoolSize = ASTPoolSize;
    }

    setClangPath(argv[0]);

    llvm::InitLLVM X(argc, argv);

    fs::path jsonPath = fs::absolute(args::get(argIR));
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
    fixCompilationDatabase(compile_commands);
    Global.cb = getCompilationDatabaseWithASTEmit(compile_commands);
    {
        auto allFiles = Global.cb->getAllFiles();
        Global.allFiles =
            std::set<std::string>(allFiles.begin(), allFiles.end());
    }

    generateICFG(*Global.cb);

    {
        // 更新 Global.npeSuspectedSources
        // 对所有 p = foo()，把函数中没有 return NULL 语句的都删掉
        for (auto p : Global.npeSuspectedSourcesItMap) {
            const std::string &signature = p.first;
            int fid = Global.getIdOfFunction(signature);
            if (fid == -1 || Global.functionReturnsNull[fid] == false)
                for (auto it : p.second) {
                    Global.npeSuspectedSources.erase(it);
                }
        }
    }

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