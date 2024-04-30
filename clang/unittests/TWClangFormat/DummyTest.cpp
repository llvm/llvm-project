#include "FormatTestBase.h"


#define DEBUG_TYPE "braces-remover-test"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace format {
namespace test {
namespace {

    class DummyTest : public FormatTestBase {};

    TEST_F(DummyTest, Dummy) {
    
    FormatStyle Style = getTWStyle();

        verifyFormat("void\r\n"
                    "f ()\r\n"
                    "{\r\n"
                    "        int q;\r\n"
                    "\r\n"
                    "    q = 9;\r\n"
                    "}",
                    "void\n"
                    "f ()\n"
                    "{\n"
                    "        int q;\n"
                    "\n"
                    "    q = 9;\n"
                    "}",
                    Style);
    }

} // namespace
} // namespace test
} // namespace format
} // namespace clang