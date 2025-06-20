#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <set>
#include <sstream>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace clang::ast_matchers;

// Create an option category for the tool
static cl::OptionCategory MyToolCategory("function-scope-identifier options");

// Command-line option for line ranges
static cl::opt<std::string> IdentifyRange(
    "identify-scope-range",
    cl::desc("Comma-separated list of line ranges (e.g., 5-10,15-20)"),
    cl::value_desc("line-ranges"),
    cl::Required,
    cl::cat(MyToolCategory));

// Parse line range string like "5-10,15-20" into vector of pairs
std::vector<std::pair<unsigned, unsigned>> parseRanges(const std::string &rangeStr) {
    std::vector<std::pair<unsigned, unsigned>> ranges;
    std::stringstream ss(rangeStr);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto dash = part.find('-');
        if (dash != std::string::npos) {
            unsigned start = std::stoi(part.substr(0, dash));
            unsigned end = std::stoi(part.substr(dash + 1));
            ranges.emplace_back(start, end);
        }
    }
    return ranges;
}

// AST matcher callback
class FunctionVisitor : public MatchFinder::MatchCallback {
    SourceManager *SM;
    std::vector<std::pair<unsigned, unsigned>> TargetRanges;

public:
    FunctionVisitor(SourceManager *SM, std::vector<std::pair<unsigned, unsigned>> Ranges)
        : SM(SM), TargetRanges(Ranges) {}

    void run(const MatchFinder::MatchResult &Result) override {
        const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
        if (!FD || !FD->hasBody()) return;

        SourceLocation startLoc = FD->getBeginLoc();
        SourceLocation endLoc = FD->getEndLoc();

        unsigned startLine = SM->getSpellingLineNumber(startLoc);
        unsigned endLine = SM->getSpellingLineNumber(endLoc);

        for (auto &[rangeStart, rangeEnd] : TargetRanges) {
            if (rangeEnd < startLine || rangeStart > endLine) continue;

            llvm::outs() << "Range " << rangeStart << "-" << rangeEnd << ":\n";
            llvm::outs() << "Function: " << FD->getNameInfo().getName().getAsString() << "\n";
            llvm::outs() << "Start Line: " << startLine << "\n";
            llvm::outs() << "End Line: " << endLine << "\n\n";
        }
    }
};

// FrontendAction to wrap matchers
class ScopeFrontendAction : public ASTFrontendAction {
public:
    std::vector<std::pair<unsigned, unsigned>> Ranges;

    ScopeFrontendAction(std::vector<std::pair<unsigned, unsigned>> R) : Ranges(R) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
        auto *Finder = new MatchFinder();
        auto *Callback = new FunctionVisitor(&CI.getSourceManager(), Ranges);

        Finder->addMatcher(functionDecl(isExpansionInMainFile()).bind("func"), Callback);
        return Finder->newASTConsumer();
    }
};

// Factory to create ScopeFrontendAction with arguments
class ScopeActionFactory : public FrontendActionFactory {
    std::vector<std::pair<unsigned, unsigned>> Ranges;

public:
    ScopeActionFactory(std::vector<std::pair<unsigned, unsigned>> R) : Ranges(R) {}

    std::unique_ptr<FrontendAction> create() override {
        return std::make_unique<ScopeFrontendAction>(Ranges);
    }
};

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }

    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    std::vector<std::pair<unsigned, unsigned>> ranges = parseRanges(IdentifyRange);
    return Tool.run(new ScopeActionFactory(ranges));
}
  