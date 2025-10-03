#include "MDParser.h"
#include "ClangDocTest.h"

namespace clang {
namespace doc {
TEST(MDParserTest, Strong) {
  MarkdownParser Parser;
  std::vector<SmallString<64>> Line = {{"**Strong**"}};
  auto Result = Parser.render(Line);
  std::string Expected = R"raw(<strong>Strong</strong>)raw";
  EXPECT_EQ(Expected, Result);
}

// TEST(MDParserTest, DoubleStrong) {
//   MarkdownParser Parser;
//   std::vector<SmallString<64>> Line = {{"****Strong****"}};
//   auto Result = Parser.render(Line);
//   std::string Expected = R"raw(<strong><strong>Strong</strong></strong>)raw";
//   EXPECT_EQ(Expected, Result);
// }

TEST(MDParserTest, Emphasis) {
  MarkdownParser Parser;
  std::vector<SmallString<64>> Line = {{"*Emphasis*"}};
  auto Result = Parser.render(Line);
  std::string Expected = R"raw(<em>Emphasis</em>)raw";
  EXPECT_EQ(Expected, Result);
}

// TEST(MDParserTest, Text) {
//   MarkdownParser Parser;
//   std::vector<SmallString<64>> Line = {{"Text"}};
//   auto Result = Parser.render(Line);
//   std::string Expected = R"raw(Text)raw";
//   EXPECT_EQ(Expected, Result);
// }
} // namespace doc
} // namespace clang
