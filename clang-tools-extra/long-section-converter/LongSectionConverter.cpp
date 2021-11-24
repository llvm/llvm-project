#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/MC/MCSectionMachO.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/MC/MCSection.h"


using namespace llvm;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...\n");

static uint16_t N;
static std::vector<std::string> HashedSectionNames;
static std::unordered_map<std::string,std::string> RenamedSectionsForward;
static std::unordered_map<std::string,std::string> RenamedSectionsReverse;

static std::string hashLongSectionName(const std::string& name) {
  return MCSection::hashLongSectionName(StringRef(name));
}

// FIXME: TargetInfo does not seem to include target object file format.
// What is the best way to determine if the target uses Mach-O?
#include <sys/utsname.h>
static StringRef getActualSectionName(const StringRef& Name) {
  //TargetInfo& TI = getContext().getTargetInfo();
  struct utsname utsn;

  ::uname(&utsn);

  if (std::string(&utsn.sysname[0]) != "Darwin") {
    return Name;
  }

  StringRef Segment;
  StringRef Section;
  unsigned unused1;
  bool unused2;
  unsigned unused3;

  (void) Segment;
  (void) unused1;
  (void) unused2;
  (void) unused3;
  auto err = MCSectionMachO::ParseSectionSpecifier(Name, Segment, Section, unused1, unused2, unused3);
  std::cout << "Segment: " << Segment.str() << " Section: " << Section.str() << std::endl;
  assert(!err && "MCSectionMachO::ParseSectionSpecifier() failed");

  return Section;
}

static void replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return;
    str.replace(start_pos, from.length(), to);
}

class LongSectionConverter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) override {
    if (const auto *node = Result.Nodes.getNodeAs<Decl>("long_section_name")) {
      
      node->dump();

      for(auto a: node->attrs()) {
        if (a->getKind() == attr::Section) {
          auto sa = static_cast<SectionAttr*>(a);
          StringRef Name = sa->getName();
          StringRef Section = getActualSectionName(Name);
          // Name can include a segment too - e.g. __DATA,__section
          auto from = Name.str();

          if (Section.size() <= N) {
            continue;
          }

          if (RenamedSectionsForward.end() == RenamedSectionsForward.find(from)) {
            if (Name == Section) {
              RenamedSectionsForward[from] = hashLongSectionName(Name.str());
            } else {
              // FIXME: should check that Section appears in Name only once
              auto tmp = Name.str();
              replace(tmp, Section.str(), hashLongSectionName(Section.str()));
              RenamedSectionsForward[from] = tmp;
            }
            
            std::string& to = RenamedSectionsForward[from];
            RenamedSectionsReverse[to] = from;
            sa->setName(*Result.Context, StringRef(to));
          } else {
            std::string& to = RenamedSectionsForward[Name.str()];
            sa->setName(*Result.Context, StringRef(to));
          }

          std::cout << from << " => " << sa->getName().str() << std::endl;
        }
      }
    }
  }
};

static std::vector<std::string> args(int argc, const char **argv) {
  std::vector<std::string> r;

  for (int i = 0; i < argc; ++i) {
    r.push_back(std::string(argv[i]));
  }

  return r;
}

int main(int argc, const char **argv) {
  // N initialized to 0
  // FIXME: it is not at all obvious how to get CodeGenOptions here
  // getASTContext() does not provide getCodeGenOptions(), but it does
  // provide getLangOptions(). Perhaps LangOptions should be used instead?
  const std::string myarg("-fhash-long-section-names=");
  for (auto& arg: args(argc, argv)) {
    // "starts_with()"
    if (arg.rfind(myarg, 0) == 0 && arg.size() > myarg.size()) {
        N = stoi(arg.substr(myarg.size()));
    }
  }

  if (N == 0) {
    std::cout << "Please add argument: " << myarg << "N" << std::endl;
  }

  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser& OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  DeclarationMatcher LongSectionMatcher = decl(hasSectionAttrWithNameLengthGreaterThan(N)).bind("long_section_name");
  LongSectionConverter Converter;
  MatchFinder Finder;

  Finder.addMatcher(LongSectionMatcher, &Converter);
  // TODO: add matchers for __asm("section$start$SEGMENT,SECTION")

  return Tool.run(newFrontendActionFactory(&Finder).get());
}