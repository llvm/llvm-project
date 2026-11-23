// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-invalid-regex-pattern %t

#include <string>

namespace llvm {
  class StringRef {
  public:
    StringRef(const char*);
    StringRef(const std::string&);
    StringRef(const std::string_view&);
  };
  
  class Regex {
  public:
    Regex(StringRef, unsigned int i = 0);
    enum RegexFlags : unsigned {
      NoFlags = 0,
      IgnoreCase = 1,
      Newline = 2,
      BasicRegex = 4,
    };
  };
} // namespace llvm

void test_detected_faulty_patterns(){
  llvm::Regex re1("(");
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid regex pattern: parentheses not balanced

  const std::string badStdString("(");
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: invalid regex pattern: parentheses not balanced
  llvm::Regex re2(badStdString);

  const char* badCharPtr = "[]";
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: invalid regex pattern: brackets ([ ]) not balanced
  llvm::Regex re3(badCharPtr);

  std::string_view badStrView("+");
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: invalid regex pattern: repetition-operator operand invalid
  llvm::Regex re4(badStrView);

  const llvm::StringRef badStrRef = "a*?";
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: invalid regex pattern: repetition-operator operand invalid
  llvm::Regex re5(badStrRef);
  
  static const char badConstStaticChar[] = "";
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: invalid regex pattern: empty (sub)expression
  llvm::Regex regex_badConstStaticChar(badConstStaticChar);

  static char badStaticChar[] = "";
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid regex pattern: empty (sub)expression
  llvm::Regex regex_badStaticChar(badStaticChar);
  
  char badCharArray[] = "";
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid regex pattern: empty (sub)expression
  llvm::Regex regex_badCharArray(badCharArray);

  struct RegexPatterns {
    const char* badMemberChar = "";
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid regex pattern: empty (sub)expression

    const std::string badMemberStr = "(";
    // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: invalid regex pattern: parentheses not balanced

    std::string_view badMemberStrView = "+";
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: invalid regex pattern: repetition-operator operand invalid

    const llvm::StringRef badMemberStrRef = "a*?";
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: invalid regex pattern: repetition-operator operand invalid

    const char badMemberConstChar[1] = "";
    // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: invalid regex pattern: empty (sub)expression
    
    char badMemberCharArray[1] = "";
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: invalid regex pattern: empty (sub)expression
  };
  
  RegexPatterns Pats;
  
  llvm::Regex re6(Pats.badMemberChar);
  llvm::Regex re7(Pats.badMemberStr);
  llvm::Regex re8(Pats.badMemberStrView);
  llvm::Regex re9(Pats.badMemberStrRef);
  llvm::Regex regex_badMemberConstStaticChar(Pats.badMemberConstChar);
  llvm::Regex regex_badMemberCharArray(Pats.badMemberCharArray);
}

void test_no_detection_on_mutable(){
  std::string badMutStdString("(");
  llvm::Regex re10(badMutStdString);

  char* badMutCharPtr = "[";
  llvm::Regex re11(badMutCharPtr);

  llvm::StringRef badMutStrRef = "a*?";
  llvm::Regex re12(badMutStrRef);

  struct RegexMutPatterns {
    char* badMutMemberChar = "";
    std::string badMutMemberStr = "(";
    llvm::StringRef badMutMemberStrRef = "a*?";
  };

  RegexMutPatterns mutPats;

  llvm::Regex re13(mutPats.badMutMemberChar);
  llvm::Regex re14(mutPats.badMutMemberStr);
  llvm::Regex re15(mutPats.badMutMemberStrRef);
}

void test_no_report_on_correct_patterns(){
  llvm::Regex re16("[0-9]");

  const std::string goodStdString("test");
  llvm::Regex re17(goodStdString);

  const char* goodCharPtr = "\\[test\\]";
  llvm::Regex re18(goodCharPtr);

  std::string_view goodStrView("testi+ng");
  llvm::Regex re19(goodStrView);

  const llvm::StringRef goodStrRef = "a*b?";
  llvm::Regex re20(goodStrRef);

  struct GoodRegexPatterns {
    const char* goodMemberChar = "[0-9]";
    const std::string goodMemberStr = "test";
    std::string_view goodMemberStrView = "\[test\]";
    const llvm::StringRef goodMemberStrRef = "a*b";
  };

  GoodRegexPatterns goodPats;

  llvm::Regex re21(goodPats.goodMemberChar);
  llvm::Regex re22(goodPats.goodMemberStr);
  llvm::Regex re23(goodPats.goodMemberStrView);
  llvm::Regex re24(goodPats.goodMemberStrRef);
}

void test_grammar_flags(){
  llvm::Regex re1_noflag("(", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid regex pattern: parentheses not balanced
  llvm::Regex re1_basic("(", 4U);
  llvm::Regex re2_basic("(", llvm::Regex::RegexFlags::BasicRegex);
}
